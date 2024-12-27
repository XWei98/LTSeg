import cv2
import os
import numpy as np
from tqdm import tqdm


path1 = '/newhome/zhaoxiaowei/Docode/LVSCiT/results/loss_2023_07_18_11_09/pred_image'
target1 = "/newhome/zhaoxiaowei/Docode/LVSCiT/results/loss_2023_07_18_11_09/mergerchannel5"
target2 = "/newhome/zhaoxiaowei/Docode/LVSCiT/results/loss_2023_07_18_11_09/mergerchannel6"
if not os.path.exists(target1):
    os.makedirs(target1)
if not os.path.exists(target2):
    os.makedirs(target2)
for path2 in tqdm(os.listdir("/newhome/zhaoxiaowei/Docode/LVSCiT/data/nd")):
    img = np.zeros([448,448,3],dtype = np.uint8)
    color = [(255,0,255),(60,255,125),(0,0,255),(125,255,60),
             (255,60,125),(255,125,60),(60,125,0),(60,0,125),
             (0,60,125),(125,60,0),(125,0,60),(0,125,60),
            (30, 30, 255), (30, 255, 30), (255, 30, 30), (30, 255, 255),
    (255, 30, 255), (255, 255, 30), (30, 30, 30), (255, 255, 255),
    (0, 128, 128), (128, 0, 128), (128, 128, 0), (64, 64, 64),
             (60,255,0),(60,0,255),(0,60,255),(255,60,0),
             (155,0,60),(0,155,60),(0,155,60),(0,155,60)]
    for i in range(29,-1,-1):
        path3 = os.path.join(path1,str(i),path2)
        #print(path3)
        img1 = cv2.imread(path3,0)
        img1 = cv2.resize(img1,(448,448))
        img1[img1<=125] = 0
        img1[img1>125] = 1

        img[:,:,0] = cv2.bitwise_or(img[:,:,0],color[i][0] * img1)
        img[:,:,1] = cv2.bitwise_or(img[:,:,1],color[i][1] * img1)
        img[:,:,2] = cv2.bitwise_or(img[:,:,2],color[i][2] * img1)

    cv2.imwrite(os.path.join(target1,path2),img)
    #print('img',path2)
    #print(os.path.join("/media/20TB/madexin3/bingbianshuju",path2))
    #print(0)
    img1 = cv2.imread(os.path.join("/newhome/zhaoxiaowei/Docode/LVSCiT/data/nd", path2), 0)

    img1 = cv2.resize(img1,(448,448))
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img =  cv2.addWeighted(img,0.5,img1,0.8,0)
    cv2.imwrite(os.path.join(target2,path2),img)
