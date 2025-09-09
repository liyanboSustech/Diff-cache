#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两张图像的频谱相似度，并可把频谱图/差异图写到磁盘
用法:
    python fft_diff.py img1 img2 --out_dir ./spectra
"""
import argparse
import sys
import numpy as np
import cv2
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# ---------- 工具函数 ----------
def load_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0

def padded_fft(gray: np.ndarray):
    h, w = gray.shape
    n_h = cv2.getOptimalDFTSize(h)
    n_w = cv2.getOptimalDFTSize(w)
    padded = cv2.copyMakeBorder(gray, 0, n_h - h, 0, n_w - w,
                                  cv2.BORDER_CONSTANT, value=0)
    fft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(fft[:, :, 0], fft[:, :, 1])
    mag = np.fft.fftshift(mag)
    mag = np.log1p(mag)          # 压缩动态范围
    return mag

def cos_sim(a: np.ndarray, b: np.ndarray):
    a, b = a.ravel(), b.ravel()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="对比两张图像的频谱相似度")
    parser.add_argument("img1", help="第一张图片路径")
    parser.add_argument("img2", help="第二张图片路径")
    parser.add_argument("--mask_low_freq", type=float, default=0.0,
                        help="仅保留中心低频比例，0~1，0 表示用全图")
    parser.add_argument("--viz", action="store_true",
                        help="弹出幅度谱对比图")
    parser.add_argument("--out_dir", default=".", help="频谱图/差异图保存目录")
    args = parser.parse_args()

    # 1. 读图
    g1, g2 = load_gray(args.img1), load_gray(args.img2)

    # 2. FFT
    m1, m2 = padded_fft(g1), padded_fft(g2)

    # 3. 可选低频 mask
    if args.mask_low_freq > 0:
        h, w = m1.shape
        cx, cy = w // 2, h // 2
        r = int(min(h, w) // 2 * args.mask_low_freq)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        m1 = cv2.bitwise_and(m1, m1, mask=mask)
        m2 = cv2.bitwise_and(m2, m2, mask=mask)

    # 4. 相似度
    sim = cos_sim(m1, m2)
    print(f"频谱余弦相似度: {sim:.4f}")

    # 5. 保存频谱图与差异图
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    m1_u8 = cv2.normalize(m1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    m2_u8 = cv2.normalize(m2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(str(out_dir / "mag1.png"), m1_u8)
    cv2.imwrite(str(out_dir / "mag2.png"), m2_u8)

    diff = cv2.normalize(np.abs(m1 - m2), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(str(out_dir / "mag_diff.png"), diff)

    # 6. 可选可视化
    if args.viz:
        cv2.imshow("mag1", m1_u8)
        cv2.imshow("mag2", m2_u8)
        cv2.imshow("mag_diff", diff)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()