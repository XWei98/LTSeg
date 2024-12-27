import os
import cv2
import numpy as np
import torch
import random
import Augmentor
import straug
from torchvision import transforms
from cProfile import label
from PIL import Image
from torch.utils.data.dataset import Dataset
from straug.blur import GaussianBlur, MotionBlur, DefocusBlur, GlassBlur, ZoomBlur
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from straug.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from straug.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color
from straug.warp import Curve, Distort, Stretch
from straug.weather import Fog, Snow, Frost, Rain, Shadow
from utils.utils import cvtColor, preprocess_input
from bert_embedding import BertEmbedding


def erasing(image):
    image = Image.fromarray(image)
    w, h = image.size

    w_occlusion_max = int(w * 0.5)
    h_occlusion_max = int(h * 0.5)

    w_occlusion_min = int(w * 0.1)
    h_occlusion_min = int(h * 0.1)

    w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
    h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

    if len(image.getbands()) == 1:
        rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 1))
    else:
        rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, len(image.getbands())) * 1))

    random_position_x = random.randint(0, w - w_occlusion)
    random_position_y = random.randint(0, h - h_occlusion)

    image.paste(rectangle, (random_position_x, random_position_y))
    image = np.asarray(image)

    return image


def straug_auto(image, prob):
    n = 3
    rng = np.random.default_rng()
    ops = []
    # ops.extend([Distort(rng)])
    ops.extend([GaussianNoise(rng), ShotNoise(rng), ImpulseNoise(rng), SpeckleNoise(rng)])
    # ops.extend([GaussianBlur(rng), MotionBlur(rng), DefocusBlur(rng), GlassBlur(rng), ZoomBlur(rng)])
    ops.extend([Contrast(rng), Brightness(rng)])
    ops.extend([Fog(rng), Snow(rng), Frost(rng), Rain(rng), Shadow(rng)])
    ops.extend([Posterize(rng), Invert(rng), Equalize(rng)])
    # ops.extend([Invert(rng)])
    image = Image.fromarray(image)
    augment = np.random.choice(ops, n)
    for op in augment:
        image = op(image, mag=np.random.randint(-1, 3), prob=prob)
    image = np.asarray(image)
    return image


class Distort:
    def __init__(self, rng=1):
        self.rng = np.random.default_rng(rng)
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = np.asarray(img)
        srcpt = []
        dstpt = []

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0
        # frac = 0.4

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        # top pts
        srcpt.append([p, p])
        x = self.rng.uniform(0, frac) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([p + x, p + y])

        srcpt.append([p + w_33, p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([p + w_33 + x, p + y])

        srcpt.append([p + w_66, p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([p + w_66 + x, p + y])

        srcpt.append([w - p, p])
        x = self.rng.uniform(-frac, 0) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([w - p + x, p + y])

        # bottom pts
        srcpt.append([p, h - p])
        x = self.rng.uniform(0, frac) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([p + x, h - p + y])

        srcpt.append([p + w_33, h - p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([p + w_33 + x, h - p + y])

        srcpt.append([p + w_66, h - p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([p + w_66 + x, h - p + y])

        srcpt.append([w - p, h - p])
        x = self.rng.uniform(-frac, 0) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([w - p + x, h - p + y])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        return img


def augmentationimage(jpgs, labels):
    for i in range(jpgs.shape[0]):
        jpg = jpgs[i, 0, :, :]
        r_move_x = random.randint(-20, 20)
        r_move_y = random.randint(-20, 20)
        r_rotate_angle = random.randint(-10, 10)  # 旋转方向取（-10，10）中的随机整数值，正为逆时针，负为顺势针
        m_move = np.float32([[1, 0, r_move_x], [0, 1, r_move_y]])  # 生成位移矩阵
        seed = random.randint(0, 1000000)
        jpg = erasing(jpg)  # erasing image for 3 times
        jpg = erasing(jpg)  # erasing image for 3 times
        jpg = erasing(jpg)  # erasing image for 3 times
        jpg = cv2.warpAffine(jpg, m_move, (jpg.shape[0], jpg.shape[1]))  # 图像位移
        center = (224 + r_move_x, 224 + r_move_y)  # 绕位移后的图片中心进行旋转
        scale = random.uniform(0.85, 1)  # 将图像缩放为100%
        m_rotate = cv2.getRotationMatrix2D(center, r_rotate_angle, scale)  # 生成旋转矩阵
        jpg = cv2.warpAffine(jpg, m_rotate, (jpg.shape[0], jpg.shape[1]))  # 图像旋转
        jpg = Image.fromarray(jpg)
        jpg = Distort(seed)(jpg)
        jpg = np.asarray(jpg)
        jpg = straug_auto(jpg, 0.5)
        jpgs[i, 0, :, :] = jpg
        jpgs[i, 1, :, :] = jpg
        jpgs[i, 2, :, :] = jpg
        for j in range(labels.shape[1]):
            label = labels[i, j, :, :].copy()
            label = cv2.warpAffine(label, m_move, (label.shape[0], label.shape[1]))  # 图像位移
            label = cv2.warpAffine(label, m_rotate, (label.shape[0], label.shape[1]))  # 图像旋转
            label = Image.fromarray(label)
            label = Distort(seed)(label)
            label = np.asarray(label)
            labels[i, j, :, :] = label
    return jpgs, labels


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        #self.dataset_path = '/data1/Datasets/Seg/Ribseg/images'
        self.dataset_path = "Datasets/img/"
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # 将图像变成PIL格式    输入为[H, W, C]输出为[H, W, C]
            # transforms.Resize(256),  # 把图片resize到给定的尺寸
            # transforms.RandomCrop(224),  # 以输入图的随机位置为中心做指定size的裁剪操作
            # transforms.Resize(self.input_shape),
            # transforms.RandomHorizontalFlip(),  # 以0.5概率水平翻转给定的PIL图像
            transforms.ToTensor(),  # 将PIL图像转换为tensor    输入为[H, W, C]输出为[C, H, W]
            # transforms.Normalize((0.485, 0.456, 0.406),
            #                      (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        r_move_x = random.randint(-20, 20)
        r_move_y = random.randint(-20, 20)
        r_rotate_angle = random.randint(-10, 10)  # 旋转方向取（-10，10）中的随机整数值，正为逆时针，负为顺势针
        m_move = np.float32([[1, 0, r_move_x], [0, 1, r_move_y]])  # 生成位移矩阵
        random_flag_erasing = random.uniform(0, 1)
        random_flag_move = random.uniform(0, 1)
        random_flag_rotate = random.uniform(0, 1)
        random_flag_distota = random.uniform(0, 1)
        random_flag_str = random.uniform(0, 1)
        seed = random.randint(0, 1000000)
        # -------------------------------#
        #   从文件中读取图像
        # -------------------------------#
        jpg = cv2.imread(os.path.join(self.dataset_path, name), 0)
        #jpg = cv2.imread(os.path.join(self.dataset_path, name + ".jpg"), 0)
        jpg = cv2.resize(jpg, self.input_shape)  # [448, 448]
        if random_flag_erasing > 0.5:
            jpg = erasing(jpg)  # erasing image for 3 times
            jpg = erasing(jpg)  # erasing image for 3 times
            jpg = erasing(jpg)  # erasing image for 3 times
            # cv2.imwrite('pass.jpg', jpg)
        if random_flag_move > 0.5:
            jpg = cv2.warpAffine(jpg, m_move, (jpg.shape[0], jpg.shape[1]))  # 图像位移
        if random_flag_rotate > 0.5:
            center = (224 + r_move_x, 224 + r_move_y)  # 绕位移后的图片中心进行旋转
            scale = random.uniform(0.85, 1)  # 将图像缩放为100%
            m_rotate = cv2.getRotationMatrix2D(center, r_rotate_angle, scale)  # 生成旋转矩阵
            jpg = cv2.warpAffine(jpg, m_rotate, (jpg.shape[0], jpg.shape[1]))  # 图像旋转
        if random_flag_distota > 0.7:
            jpg = Image.fromarray(jpg)
            jpg = Distort(seed)(jpg)
            jpg = np.asarray(jpg)
        if random_flag_str > 0.5:
            jpg = straug_auto(jpg, 0.5)
        # cv2.imwrite("flag.jpg", jpg)

        jpg = np.expand_dims(jpg, -1).repeat(3, axis=-1)  # [448, 448, 3]
        jpg = self.transform(jpg)

        label_list = []
        for i in range(self.num_classes):

            label = cv2.imread(os.path.join('Dataset/labels', str(i), name), 0)
            label = cv2.resize(label, self.input_shape, interpolation=cv2.INTER_NEAREST)  # [448, 448]
            if random_flag_move > 0.5:
                label = cv2.warpAffine(label, m_move, (label.shape[0], label.shape[1]))  # 图像位移
            if random_flag_rotate > 0.5:
                label = cv2.warpAffine(label, m_rotate, (label.shape[0], label.shape[1]))  # 图像旋转
            if random_flag_distota > 0.7:
                label = Image.fromarray(label)
                label = Distort(seed)(label)
                label = np.asarray(label)
            label = label.copy()
            label[label > 125] = 255
            label[label <= 125] = 0
            label = label / 255  # 映射到0、1区间
            label_list.append(label)
        seg_labels = np.stack(label_list, axis=0)

        ovelaplabel_list = []
        for i in range(self.num_classes):
            label = cv2.imread(os.path.join('Dataset/label/overlap', str(i), name), 0)

            label = cv2.resize(label, self.input_shape, interpolation=cv2.INTER_NEAREST)  # [448, 448]
            if random_flag_move > 0.5:
                label = cv2.warpAffine(label, m_move, (label.shape[0], label.shape[1]))  # 图像位移
            if random_flag_rotate > 0.5:
                label = cv2.warpAffine(label, m_rotate, (label.shape[0], label.shape[1]))  # 图像旋转
            if random_flag_distota > 0.7:
                label = Image.fromarray(label)
                label = Distort(seed)(label)
                label = np.asarray(label)
            label = label.copy()
            label[label > 125] = 255
            label[label <= 125] = 0
            label = label / 255
            ovelaplabel_list.append(label)
        ovelapseg_labels = np.stack(label_list, axis=0)

        nonovelaplabel_list = []
        for i in range(self.num_classes):
            label = cv2.imread(os.path.join('Dataset/label/nonoverlap',str(i), name), 0)

            label = cv2.resize(label, self.input_shape, interpolation=cv2.INTER_NEAREST)  # [448, 448]
            if random_flag_move > 0.5:
                label = cv2.warpAffine(label, m_move, (label.shape[0], label.shape[1]))  # 图像位移
            if random_flag_rotate > 0.5:
                label = cv2.warpAffine(label, m_rotate, (label.shape[0], label.shape[1]))  # 图像旋转
            if random_flag_distota > 0.7:
                label = Image.fromarray(label)
                label = Distort(seed)(label)
                label = np.asarray(label)
            label = label.copy()
            label[label > 125] = 255
            label[label <= 125] = 0
            label = label / 255
            nonovelaplabel_list.append(label)
        nonovelapseg_labels = np.stack(label_list, axis=0)


        return jpg, seg_labels,ovelapseg_labels,nonovelapseg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(label)
        # label   = Image.fromarray(np.array(label))
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images = []
    pngs = []
    overpngs = []
    nonoverpngs = []

    for img, png, overpng,nonoverpng in batch:
        # images.append(img)      # 不使用归一化代码时
        images.append(img.numpy())
        pngs.append(png)
        overpngs.append(overpng)
        nonoverpngs.append(nonoverpng)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    overpngs = torch.from_numpy(np.array(overpngs)).long()

    nonoverpngs = torch.from_numpy(np.array(nonoverpngs)).long()


    return images, pngs, overpngs, nonoverpngs
