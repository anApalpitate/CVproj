#!/usr/bin/python3

import numpy as np
import numpy.fft as fft

from utils import (
    load_image,
    save_image,
    PIL_image_to_numpy_arr,
    rgb2gray,
    numpy_arr_to_PIL_image
)

def freqCompress(image: np.ndarray, ratio: float) -> np.ndarray:
    """
    使用二维傅里叶变换和低通滤波实现频率压缩。

    参数:
        image: 输入图像（灰度图，2D numpy 数组）。
        ratio: 保留的低频成分的比例。

    返回:
        压缩后的重建图像。
    """
    # 进行二维傅里叶变换
    f_transform = np.fft.fftshift(np.fft.fft2(image))
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2
    cutoff_x = int(center_x * ratio)
    cutoff_y = int(center_y * ratio)

    # 创建低通滤波器掩膜
    mask = np.zeros((rows, cols))
    mask[center_x - cutoff_x:center_x + cutoff_x, center_y - cutoff_y:center_y + cutoff_y] = 1

    # 应用低通滤波器
    f_transform_filtered = f_transform * mask

    # 进行逆傅里叶变换
    f_transform_inverse = np.fft.ifft2(np.fft.ifftshift(f_transform_filtered))

    # 返回逆傅里叶变换的实部（压缩后的图像）
    compressed_image = np.abs(f_transform_inverse)
    return compressed_image

def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    计算原始图像与压缩图像之间的峰值信噪比（PSNR）。

    参数:
        original: 原始图像。
        compressed: 压缩图像。

    返回:
        PSNR 值。
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')  # 如果没有误差，返回无穷大 PSNR
    return 20 * np.log10(1.0 / np.sqrt(mse))

def part4():
    image = load_image("../assets/steam.jpg")

    ratios = [0.1, 0.3, 0.5, 0.7]
    image_np = PIL_image_to_numpy_arr(image)
    if image_np.ndim == 3:
        image_np = rgb2gray(image_np)

    for ratio in ratios:
        compressedImage = freqCompress(image_np, ratio)

        psnr = calculate_psnr(image_np, compressedImage)
        print(f"[INFO] ratio: {ratio:.1f} | PSNR: {psnr:.2f} dB")

        save_path = f"../results/part4/compressed_of_ratio{ratio:.1f}.jpg"
        numpy_arr_to_PIL_image(compressedImage, scale_to_255=True)
        save_image(save_path, compressedImage)
        print(f"保存压缩图像到: {save_path}\n")

if __name__ == "__main__":
    part4()
