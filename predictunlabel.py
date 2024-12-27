import os
import cv2
import torch
import numpy as np

from torch import nn
from tqdm import tqdm
from torchvision import transforms
import argparse
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

import scipy
from scipy.spatial.distance import directed_hausdorff, cdist


if __name__ == "__main__":
    num_classes = 24
    backbone = 'resnet50'
    input_shape = (448, 448)
    dataset_paths = [
        '/jsrt/',
        '/nih/',
        '/shenzhen/'
    ]
    path = "/data1/Datasets/Seg/openxray"
    for i in range(len(dataset_paths)):
        image_dir = path + dataset_paths[i]
        #image_dir = '/data1/Code/zhaoxiaowei/LVSCiTLos/data/vxraytest'
        save_pdir = '/data1/Code/zhaoxiaowei/LVSCiTdecople/results_medx/Decople_hard_pretrain_all/12_con_ma(a+b)_71.18_81.29/ModelLabel2024_07_15_15_07'
        modelsort = "/last"
        model_path = save_pdir + modelsort +'_epoch_weights.pth'


        pred_save_path = save_pdir + modelsort + dataset_paths[i]
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
        # unet  = AttU_Net(num_classes = num_classes)
        # unet = UNetPuls(num_classes = num_classes,supervised=False)

        # unet = swinunetr(img_size=448, in_channels=3, out_channels=num_classes, )
        # ucconfig_vit = ucconfig.get_CTranS_config()
        # unet = UCTransNet(ucconfig_vit, n_channels=ucconfig.n_channels, n_classes=num_classes)
        # unet = UNext(num_classes = num_classes)
        # unet = swinunet(num_classes = num_classes)

        # unet = VMUNet(num_classes=20,input_channels=3,depths=[2, 2, 2, 2],depths_decoder=[2, 2, 2, 1],drop_path_rate=0.2,load_ckpt_path=model_path)
        # model.load_from()
        # unet = UNet(n_classes = num_classes)
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

            pred = unet(image)      # [b, num_classes, h, w]


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
