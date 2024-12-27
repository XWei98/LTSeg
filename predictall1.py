import os
import cv2
import torch
import numpy as np

from torch import nn
from tqdm import tqdm
from torchvision import transforms
import argparse
from networks.vit_seg_modeling import SCNet as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

import scipy
from scipy.spatial.distance import directed_hausdorff, cdist



def calmdsc(imgdir1, imgdir2):  # mdice
    miou = 0
    for img in os.listdir(imgdir1):
        imgpath1 = os.path.join(imgdir1, img)
        imgpath2 = os.path.join(imgdir2, img)
        img1 = cv2.imread(imgpath1, 0)
        img2 = cv2.imread(imgpath2, 0)
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1
        img2[img2 <= 125] = 0
        img2[img2 > 125] = 1
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        img3 = cv2.bitwise_and(img1, img2)
        img4 = cv2.bitwise_or(img1, img2)
        iou = 2 * img3.ravel().sum() / (img4.ravel().sum() + img3.ravel().sum()) if (
                                                                                                img4.ravel().sum() + img3.ravel().sum()) != 0 else 0
        miou = miou + iou
    return miou / len(os.listdir(imgdir1))


def calmiou(imgdir1, imgdir2):  # miou
    miou = 0
    mdsc = 0
    for img in os.listdir(imgdir1):
        imgpath1 = os.path.join(imgdir1, img)
        imgpath2 = os.path.join(imgdir2, img)
        img1 = cv2.imread(imgpath1, 0)
        img2 = cv2.imread(imgpath2, 0)
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1
        img2[img2 <= 125] = 0
        img2[img2 > 125] = 1
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        img3 = cv2.bitwise_and(img1, img2)
        img4 = cv2.bitwise_or(img1, img2)
        iou = img3.ravel().sum() / img4.ravel().sum() if img4.ravel().sum() != 0 else 0
        miou = miou + iou

        dsc = 2 * img3.ravel().sum() / (img4.ravel().sum() + img3.ravel().sum()) if (
                                                                                                img4.ravel().sum() + img3.ravel().sum()) != 0 else 0
        mdsc = mdsc + dsc

    return miou / len(os.listdir(imgdir1)), mdsc / len(os.listdir(imgdir1))


def calall(imgdir1, imgdir2):
    # 初始化各项指标
    miou = mdsc = accuracy = specificity = sensitivity = 0
    total_imgs = len(os.listdir(imgdir1))  # 计算图像总数

    for img in os.listdir(imgdir1):
        imgpath1 = os.path.join(imgdir1, img)
        imgpath2 = os.path.join(imgdir2, img)
        img1 = cv2.imread(imgpath1, 0)
        img2 = cv2.imread(imgpath2, 0)
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1
        img2[img2 <= 125] = 0
        img2[img2 > 125] = 1
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)

        # 计算交并比(IoU)和Dice相似系数(DSC)
        img3 = cv2.bitwise_and(img1, img2)
        img4 = cv2.bitwise_or(img1, img2)
        iou = img3.ravel().sum() / img4.ravel().sum() if img4.ravel().sum() != 0 else 0
        miou += iou
        dsc = 2 * img3.ravel().sum() / (img4.ravel().sum() + img3.ravel().sum()) if (
                                                                                                img4.ravel().sum() + img3.ravel().sum()) != 0 else 0
        mdsc += dsc

        # 计算TP, TN, FP, FN
        tp = np.logical_and(img1 == 1, img2 == 1).sum()
        tn = np.logical_and(img1 == 0, img2 == 0).sum()
        fp = np.logical_and(img1 == 0, img2 == 1).sum()
        fn = np.logical_and(img1 == 1, img2 == 0).sum()

        # 计算准确性(Accuracy)，特异性(Specificity)，敏感性(Sensitivity)
        accuracy += (tp + tn) / (tp + tn + fp + fn)
        specificity += tn / (tn + fp) if (tn + fp) != 0 else 0
        sensitivity += tp / (tp + fn) if (tp + fn) != 0 else 0

    # 计算平均值并返回
    return miou / total_imgs, mdsc / total_imgs, accuracy / total_imgs, specificity / total_imgs, sensitivity / total_imgs


def calalll(imgdir1, imgdir2):
    total_imgs = len(os.listdir(imgdir1))
    if total_imgs == 0:
        return 0, 0, 0, 0, 0, 0, 0  # 如果没有图像，返回0

    miou = mdsc = accuracy = specificity = sensitivity = hausdorff_distance = assd = 0

    for img in os.listdir(imgdir1):
        imgpath1 = os.path.join(imgdir1, img)
        imgpath2 = os.path.join(imgdir2, img)
        img1 = cv2.imread(imgpath1, 0)
        img2 = cv2.imread(imgpath2, 0)
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1
        img2[img2 <= 125] = 0
        img2[img2 > 125] = 1
        img1 = img1.astype(np.bool_)
        img2 = img2.astype(np.bool_)

        # 计算交并比(IoU)和Dice相似系数(DSC)
        img3 = img1 & img2
        img4 = img1 | img2
        iou = img3.sum() / img4.sum() if img4.sum() != 0 else 0
        miou += iou
        dsc = 2 * img3.sum() / (img4.sum() + img3.sum()) if (img4.sum() + img3.sum()) != 0 else 0
        mdsc += dsc

        # 计算准确性，特异性，敏感性
        tp = (img1 & img2).sum()
        tn = (~img1 & ~img2).sum()
        fp = (~img1 & img2).sum()
        fn = (img1 & ~img2).sum()
        accuracy += (tp + tn) / (tp + tn + fp + fn)
        specificity += tn / (tn + fp) if (tn + fp) != 0 else 0
        sensitivity += tp / (tp + fn) if (tp + fn) != 0 else 0

        # 计算Hausdorff距离
        y_true, x_true = np.where(img1)
        y_pred, x_pred = np.where(img2)
        if y_true.size and y_pred.size:  # 确保图像不全是黑色的
            hausdorff_dist = max(
                directed_hausdorff(np.stack([y_true, x_true], axis=1), np.stack([y_pred, x_pred], axis=1))[0],
                directed_hausdorff(np.stack([y_pred, x_pred], axis=1), np.stack([y_true, x_true], axis=1))[0])
        else:
            hausdorff_dist = 0
        hausdorff_distance += hausdorff_dist

        # 计算平均对称表面距离(ASSD)
        if y_true.size and y_pred.size:  # 确保图像不全是黑色的
            true_points = np.stack([y_true, x_true], axis=1)
            pred_points = np.stack([y_pred, x_pred], axis=1)
            dist_matrix = scipy.spatial.distance.cdist(true_points, pred_points, 'euclidean')
            assd += np.mean(np.min(dist_matrix, axis=0)) + np.mean(np.min(dist_matrix, axis=1))
        else:
            assd += 0

    return miou / total_imgs, mdsc / total_imgs, accuracy / total_imgs, specificity / total_imgs, sensitivity / total_imgs, hausdorff_distance / total_imgs, assd / (
                2 * total_imgs)


def cal_metrics(imgdir1, imgdir2):  # miou, mdsc, sensitivity, specificity, precision, recall
    miou = 0
    mdsc = 0
    sensitivity = 0
    specificity = 0
    precision = 0
    recall = 0
    hausdorff_distance = 0
    assd = 0
    total_imgs = len(os.listdir(imgdir1))  # 计算图像总数
    for img in os.listdir(imgdir1):
        imgpath1 = os.path.join(imgdir1, img)
        imgpath2 = os.path.join(imgdir2, img)
        img1 = cv2.imread(imgpath1, 0)
        img2 = cv2.imread(imgpath2, 0)
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1
        img2[img2 <= 125] = 0
        img2[img2 > 125] = 1
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)

        img3 = cv2.bitwise_and(img1, img2)
        img4 = cv2.bitwise_or(img1, img2)

        # Intersection over Union
        iou = img3.ravel().sum() / img4.ravel().sum() if img4.ravel().sum() != 0 else 0
        miou = miou + iou

        # Dice Similarity Coefficient
        dsc = 2 * img3.ravel().sum() / (img1.ravel().sum() + img2.ravel().sum()) if (
                                                                                            img1.ravel().sum() + img2.ravel().sum()) != 0 else 0
        mdsc = mdsc + dsc

        # True Positives, False Positives, True Negatives, False Negatives
        TP = img3.ravel().sum()
        FP = img2.ravel().sum() - TP
        FN = img1.ravel().sum() - TP
        TN = img1.size - (TP + FP + FN)

        # Sensitivity / Recall
        recall_img = TP / (TP + FN) if (TP + FN) != 0 else 0
        recall += recall_img

        # Specificity
        specificity_img = TN / (TN + FP) if (TN + FP) != 0 else 0
        specificity += specificity_img

        # Precision
        precision_img = TP / (TP + FP) if (TP + FP) != 0 else 0
        precision += precision_img

        # Sensitivity
        sensitivity_img = recall_img  # Sensitivity is the same as recall
        sensitivity += sensitivity_img
        # 计算Hausdorff距离
        y_true, x_true = np.where(img1)
        y_pred, x_pred = np.where(img2)
        if y_true.size and y_pred.size:  # 确保图像不全是黑色的
            hausdorff_dist = max(
                directed_hausdorff(np.stack([y_true, x_true], axis=1), np.stack([y_pred, x_pred], axis=1))[0],
                directed_hausdorff(np.stack([y_pred, x_pred], axis=1), np.stack([y_true, x_true], axis=1))[0])
        else:
            hausdorff_dist = 0
        hausdorff_distance += hausdorff_dist

        # 计算平均对称表面距离(ASSD)
        if y_true.size and y_pred.size:  # 确保图像不全是黑色的
            true_points = np.stack([y_true, x_true], axis=1)
            pred_points = np.stack([y_pred, x_pred], axis=1)
            dist_matrix = scipy.spatial.distance.cdist(true_points, pred_points, 'euclidean')
            assd += np.mean(np.min(dist_matrix, axis=0)) + np.mean(np.min(dist_matrix, axis=1))
        else:
            assd += 0

    num_images = len(os.listdir(imgdir1))

    return miou / num_images, mdsc / num_images, sensitivity / num_images, specificity / num_images, precision / num_images, recall / num_images, hausdorff_distance / total_imgs, assd / (
                2 * total_imgs)


if __name__ == "__main__":
    num_classes = 20
    backbone = 'resnet50'
    input_shape = (448, 448)

    image_dir = '/data1/test'
    save_pdir = '/ModelLabel'
    modelsort = "/best"
    model_path = save_pdir + modelsort +'_epoch_weights.pth'


    pred_save_path = save_pdir + modelsort + '/pred_image'
    miou_save = save_pdir + modelsort + '/pred_miou.txt'
    mdsc_save = save_pdir + modelsort + '/pred_mdice.txt'
    mspec_save = save_pdir + modelsort + '/pred_mspec.txt'
    mprec_save = save_pdir + modelsort + '/pred_mprec.txt'
    mrecall_save = save_pdir + modelsort + '/pred_mrecall.txt'
    mhd_save = save_pdir + modelsort + '/pred_mhd.txt'
    massd_save = save_pdir + modelsort + '/pred_assd.txt'
    msen_save = save_pdir + modelsort + '/pred_msen.txt'

    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将图像变成PIL格式    输入为[H, W, C]输出为[H, W, C]
        transforms.ToTensor(),  # 将PIL图像转换为tensor    输入为[H, W, C]输出为[C, H, W]
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')  # 突触
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=30, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=448, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    args = parser.parse_args()
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    config_vit.n_patches = int(args.img_size / args.vit_patches_size) * int(args.img_size / args.vit_patches_size)
    config_vit.h = int(args.img_size / args.vit_patches_size)
    config_vit.w = int(args.img_size / args.vit_patches_size)
    unet = ViT_seg(config_vit, img_size=448, num_classes=num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # 加载模型参数
    unet = unet.eval()  # 测试模式
    unet = nn.DataParallel(unet)
    unet = unet.cuda()

    print(
        '==============================================Predicted Image Save!==============================================')
    for img in tqdm(os.listdir(image_dir)):
        imgpath = os.path.join(image_dir, img)
        image = cv2.imread(imgpath, 0)
        image = cv2.resize(image, (448, 448))

        # 无归一化使用
        '''
        image = np.expand_dims(image, 0).repeat(3, axis=0)    # [3, 448, 448]
        image = np.expand_dims(image, 0)                      # [b, 3, 448, 448]
        image = torch.from_numpy(image).type(torch.FloatTensor)
        '''

        # 归一化使用
        image = np.expand_dims(image, -1).repeat(3, axis=-1)  # [448, 448, 3]
        image = transform(image)  # [3, 448, 448]
        image = image.unsqueeze(0)  # [b, 3, 448, 448]

        pred,overpred,nonover ,predx = unet(image)      # [b, num_classes, h, w]
        pred = pred+overpred+nonover+predx

        pred = pred.detach().cpu().numpy()
        # pred = t_crf(image.cpu().numpy(), pred)     # 后处理CRF

        for i in range(num_classes):
            save_path = os.path.join(pred_save_path, str(i))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            savepath = os.path.join(save_path, img)  # 拼接存储路径

            pred_image = pred[0, i, :, :]
            pred_image = pred_image * 255
            pred_image[pred_image <= 127] = 0
            pred_image[pred_image > 127] = 255
            pred_image = pred_image.astype(np.uint8)
            cv2.imwrite(savepath, pred_image)
    print(
        '==================================================Compute  MIOU==================================================')
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
               '19',]#'20', '21', '22', '23','24', '25', '26','27', '28', '29','30', '31'
    iou_list = []
    iou_str_list = []
    dsc_list = []
    dsc_str_list = []
    macc_list = []
    macc_str_list = []
    msen_list = []
    msen_str_list = []
    mspec_list = []
    mspec_str_list = []
    mprec_list = []
    mprec_str_list = []
    mrecall_list = []
    mrecall_str_list = []
    mhd_list = []
    mhd_str_list = []
    massd_list = []
    massd_str_list = []

    for cls in tqdm(classes):
        imgpath = os.path.join(pred_save_path, cls)

        labelpath = os.path.join('/Datasets/labels', cls)


        miou, mdsc, msen, mspec, mprec, mrecall, mhd, massd = cal_metrics(imgpath, labelpath)
        iou_list.append(miou)
        miou_str = '第{}标签的miou: {}'.format(str(int(cls) + 1), miou)
        iou_str_list.append(miou_str)

        dsc_list.append(mdsc)
        mdsc_str = '第{}标签的mdsc: {}'.format(str(int(cls) + 1), mdsc)
        dsc_str_list.append(mdsc_str)

        msen_list.append(msen)
        msen_str = '第{}标签的msens: {}'.format(str(int(cls) + 1), msen)
        msen_str_list.append(msen_str)

        mspec_list.append(mspec)
        mspec_str = '第{}标签的mspec: {}'.format(str(int(cls) + 1), mspec)
        mspec_str_list.append(mspec_str)

        mprec_list.append(mprec)
        mprec_str = '第{}标签的mprec: {}'.format(str(int(cls) + 1), mprec)
        mprec_str_list.append(mprec_str)

        mrecall_list.append(mrecall)
        mrecall_str = '第{}标签的mrecall: {}'.format(str(int(cls) + 1), mrecall)
        mrecall_str_list.append(mrecall_str)

        mhd_list.append(mhd)
        mhd_str = '第{}标签的mhd: {}'.format(str(int(cls) + 1), mhd)
        mhd_str_list.append(mhd_str)

        massd_list.append(massd)
        massd_str = '第{}标签的massd: {}'.format(str(int(cls) + 1), massd)
        massd_str_list.append(massd_str)

    mmiou_str = sum(iou_list) / len(iou_list)
    miou_str = '\n'.join(iou_str_list) + '\n平均miou指标：' + str(mmiou_str)
    print('平均指标：', mmiou_str)
    with open(miou_save, 'w') as f:
        f.write(miou_str)

    mmdsc_str = sum(dsc_list) / len(dsc_list)
    mdsc_str = '\n'.join(dsc_str_list) + '\n平均mdsc指标：' + str(mmdsc_str)
    print('平均指标：', mmdsc_str)
    with open(mdsc_save, 'w') as f:
        f.write(mdsc_str)

    mmsen_str = sum(msen_list) / len(msen_list)
    msen_str = '\n'.join(msen_str_list) + '\n平均msens指标：' + str(mmsen_str)
    print('平均指标：', mmsen_str)
    with open(msen_save, 'w') as f:
        f.write(msen_str)

    mmspec_str = sum(mspec_list) / len(mspec_list)
    mspec_str = '\n'.join(mspec_str_list) + '\n平均mspec指标：' + str(mmspec_str)
    print('平均指标：', mmspec_str)
    with open(mspec_save, 'w') as f:
        f.write(mspec_str)

    mmprec_str = sum(mprec_list) / len(mprec_list)
    mprec_str = '\n'.join(mprec_str_list) + '\n平均mprec指标：' + str(mmprec_str)
    print('平均指标：', mmprec_str)
    with open(mprec_save, 'w') as f:
        f.write(mprec_str)

    mmrecall_str = sum(mrecall_list) / len(mrecall_list)
    mrecall_str = '\n'.join(mrecall_str_list) + '\n平均mrecall指标：' + str(mmrecall_str)
    print('平均指标：', mmrecall_str)
    with open(mrecall_save, 'w') as f:
        f.write(mrecall_str)

    mmhd_str = sum(mhd_list) / len(mhd_list)
    mhd_str = '\n'.join(mhd_str_list) + '\n平均mhd指标：' + str(mmhd_str)
    print('平均指标：', mmhd_str)
    with open(mhd_save, 'w') as f:
        f.write(mhd_str)

    mmassd_str = sum(massd_list) / len(massd_list)
    massd_str = '\n'.join(massd_str_list) + '\n平均massd指标：' + str(mmassd_str)
    print('平均指标：', mmassd_str)
    with open(massd_save, 'w') as f:
        f.write(massd_str)


