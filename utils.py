import numpy as np
import cv2


# 用于线性映射（linear mapping）
def linear_mapping(images):
    max_value = images.max()
    min_value = images.min()

    parameter_a = 1 / (max_value - min_value)
    parameter_b = 1 - max_value * parameter_a

    image_after_mapping = parameter_a * images + parameter_b

    return image_after_mapping


# 图像预处理
def pre_process(img):
    # 获取图像的大小
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # 使用汉宁窗
    window = window_func_2d(height, width)
    img = img * window

    return img


def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)  # 根据输入的坐标向量生成对应的坐标矩阵

    win = mask_col * mask_row

    return win


def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    # 旋转图像
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot
