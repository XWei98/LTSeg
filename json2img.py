import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os

# 标签列表
list_label = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10',
              'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10']

# 设置基础路径
annotations_base_path = 'D:/Xw_Data/Vxray/Annotations'
images_base_path = 'D:\Xw_Data/Vxray'

# 读取训练集的注释
train_annotations_path = os.path.join(annotations_base_path, 'test/Vindr_RibCXR_val_mask.json')
train_data = pd.read_json(train_annotations_path)

# 遍历训练数据中的每一项
for i in range(len(train_data)):
    # 获取当前图像的相对路径
    img_relative_path = train_data['img'][i]  # 示例："some/path/to/image.png"
    img_name = os.path.basename(img_relative_path)


    img_path = os.path.join(images_base_path, img_relative_path)


    img = Image.open(img_path)
    img = img.convert('RGB')
    img = np.asarray(img, dtype=np.uint8)

    # 对于每个标签
    for index, label_name in enumerate(list_label):
        # 初始化二值图
        label = np.zeros(img.shape[:2], dtype=np.uint8)

        # 提取标签坐标
        pts = train_data[label_name][i]
        if pts != 'None':
            pts = np.array([[[int(pt['x']), int(pt['y'])]] for pt in pts])
            label = cv2.fillPoly(label, [pts], 255)

        # 创建对应编号的文件夹
        binary_masks_dir = os.path.join(images_base_path, 'val', f'{index}')
        os.makedirs(binary_masks_dir, exist_ok=True)

        # 保存二值图
        binary_mask_path = os.path.join(binary_masks_dir, f'{os.path.splitext(img_name)[0]}_{label_name}.png')
        cv2.imwrite(binary_mask_path, label)

        print(f'二值图 {binary_mask_path} 已保存。')
