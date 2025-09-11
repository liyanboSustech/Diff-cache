import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_frequency(image_path, low_freq_ratio=0.1):
    """
    图像高低频分析
    :param image_path: 输入图像路径
    :param low_freq_ratio: 保留低频的比例 (0~1之间，值越大低频范围越大)
    """
    # 读取图像并转灰度
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法读取图像，请检查路径是否正确。")

    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # 中心化

    # 频谱幅度（对数显示）
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 构建低频掩码
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    r = int(min(crow, ccol) * low_freq_ratio)  # 半径
    mask[crow-r:crow+r, ccol-r:ccol+r] = 1

    # 低频部分
    low_freq = fshift * mask
    low_img = np.fft.ifft2(np.fft.ifftshift(low_freq))
    low_img = np.abs(low_img)

    # 高频部分
    high_freq = fshift * (1 - mask)
    high_img = np.fft.ifft2(np.fft.ifftshift(high_freq))
    high_img = np.abs(high_img)

    # 可视化
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.axis('off')

    plt.subplot(2, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Frequency Spectrum'), plt.axis('off')

    plt.subplot(2, 3, 3), plt.imshow(mask*255, cmap='gray')
    plt.title('Low Frequency Mask'), plt.axis('off')

    plt.subplot(2, 3, 4), plt.imshow(low_img, cmap='gray')
    plt.title('Low Frequency Image'), plt.axis('off')

    plt.subplot(2, 3, 5), plt.imshow(high_img, cmap='gray')
    plt.title('High Frequency Image'), plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例：替换为你的图片路径
    analyze_frequency("/home/lyb/Diff-cache/results/flux_result_dp1_cfg1_ulysses2_ring2_tp1_pp1_patchNone_0_tc_False.png", low_freq_ratio=0.1)
