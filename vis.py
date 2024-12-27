import cv2
import os
import numpy as np
from tqdm import tqdm
import os
import cv2
import numpy as np
from tqdm import tqdm

path = "/data1/Code/zhaoxiaowei/LVSCiTdecople/result_vis_medx/12_con_ma(a+b)_71.18_81.29/"
path1 = "/data1/Code/zhaoxiaowei/LVSCiTdecople/results_medx/Decople_hard_pretrain_all/12_con_ma(a+b)_71.18_81.29/ModelLabel2024_07_15_15_07/last/pred_image"
target_dirs = {
    "rib1": path1 + "/all/rib1",
    "rib2": path1 + "/all/rib2",

}
#image_dir = '/data1/Code/zhaoxiaowei/LVSCiTLos/data/vxraytest'
image_dir = "/data1/Code/zhaoxiaowei/LVSCiTdecople/data/zxwtesthard"
# 创建目标文件夹
for target in target_dirs.values():
    if not os.path.exists(target):
        os.makedirs(target)

# 定义颜色，调整RGB值使其变浅
color = [(255, 153, 255), (153, 255, 204), (153, 153, 255), (204, 255, 153),
         (255, 153, 204), (255, 204, 153), (153, 204, 153), (153, 153, 204),
         (153, 204, 255), (204, 153, 153), (204, 153, 204), (153, 255, 153),
         (102, 102, 255), (102, 255, 102), (255, 102, 102), (102, 255, 255),
         (255, 102, 255), (255, 255, 102), (102, 102, 102), (255, 255, 255),
         (102, 204, 204), (204, 102, 204), (204, 204, 102), (128, 128, 128),
         (153, 255, 102), (153, 102, 255), (102, 153, 255), (255, 153, 102),
         (204, 102, 153), (102, 204, 153)]

# 定义索引组合和对应的目标文件夹
index_combinations = {
    (24, 25): ("Clavicles1", "Clavicles2"),
    (26, 27): ("Scapulas1", "Scapulas2"),
    (28, 29): ("Lungs1", "Lungs2"),
    (30,): ("Trachea1", "Trachea2"),
    (31,): ("Mediastinum1", "Mediastinum2")


}

# 添加颜色组合，确保每组颜色不同且颜色变浅
combination_colors = {
    #(24, 25): ([(255, 255, 255), (255, 204, 204)]),
    #(26, 27): ([(153, 255, 255), (204, 255, 255)]),
    #(28, 29): ([(153, 204, 255), (204, 229, 255)]),
    #(30,): ([(153, 255, 153)]),
    #(31,): ([(204, 204, 255)])

    (24, 25): ([(255, 153, 255), (153, 255, 204),]),
    (26, 27): ([(153, 153, 255), (204, 255, 153)]),
    (28, 29): ([(255, 153, 204), (255, 204, 153)]),
    (30,): ([(153, 204, 153)]),
    (31,): ([(153, 153, 204)])
}

for imgname in tqdm(os.listdir(image_dir)):
    img = np.zeros([448, 448, 3], dtype=np.uint8)

    for i in range(23, -1, -1):
        path3 = os.path.join(path1, str(i), imgname)
        img1 = cv2.imread(path3, 0)
        img1 = cv2.resize(img1, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1

        img[:, :, 0] = cv2.bitwise_or(img[:, :, 0], color[i][0] * img1)
        img[:, :, 1] = cv2.bitwise_or(img[:, :, 1], color[i][1] * img1)
        img[:, :, 2] = cv2.bitwise_or(img[:, :, 2], color[i][2] * img1)

    # 保存普通rib图像
    cv2.imwrite(os.path.join(target_dirs["rib1"], imgname), img)

    img1 = cv2.imread(os.path.join(image_dir, imgname), 0)
    img1 = cv2.resize(img1, (448, 448))
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 0.5, img1, 0.8, 0)
    cv2.imwrite(os.path.join(target_dirs["rib2"], imgname), img)