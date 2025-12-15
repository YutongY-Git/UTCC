import cv2
import numpy as np
import os

# 输入文件夹路径
maskFile = r'E:\UTB-finish\UTB_master\MRCNNimage\mask1'
imgFile = r'E:\UTB-finish\UTB_master\MRCNNimage\img'
output_mask_folder = r"E:\UTB-finish\UTB_master\MRCNNimage\cut\mask"
output_img_folder = r"E:\UTB-finish\UTB_master\MRCNNimage\cut\img"

# 确保输出文件夹存在
os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_img_folder, exist_ok=True)

# 裁剪尺寸
hy = 321
wy = 321

# 遍历 mask 文件夹中的所有图像文件
for mask_file in os.listdir(maskFile):
    if mask_file.endswith(".png"):
        img_file = mask_file.replace(".png", ".jpg")
        mask_path = os.path.join(maskFile, mask_file)
        img_path = os.path.join(imgFile, img_file)

        if os.path.exists(mask_path) and os.path.exists(img_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # 获取原图尺寸
            h, w = mask.shape[:2]

            # 计算需要补充的大小
            pad_top = (hy - h % hy) % hy
            pad_left = (wy - w % wy) % wy

            # 补充白色像素
            mask_padded = cv2.copyMakeBorder(mask, 0, pad_top, 0, pad_left, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            img_padded = cv2.copyMakeBorder(img, 0, pad_top, 0, pad_left, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            # 更新原图尺寸
            h_p, w_p = mask_padded.shape[:2]

            # 计算裁剪数量
            h_n = h_p // hy
            w_n = w_p // wy

            for i in range(h_n):
                for j in range(w_n):
                    mask_patch = mask_padded[i * hy:(i + 1) * hy, j * wy:(j + 1) * wy]
                    img_patch = img_padded[i * hy:(i + 1) * hy, j * wy:(j + 1) * wy]

                    # 保存裁剪后的图像
                    cv2.imwrite(os.path.join(output_mask_folder, f"{mask_file.replace('.png', f'_{i:02d}{j:02d}.png')}"), mask_patch)
                    cv2.imwrite(os.path.join(output_img_folder, f"{img_file.replace('.jpg', f'_{i:02d}{j:02d}.jpg')}"), img_patch)

            print(f'处理完成: {mask_file}')
        else:
            print(f'警告: {mask_path} 或 {img_path} 不存在')

print("所有文件处理完成。")
