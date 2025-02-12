import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import cv2

def read_and_crop_image(image_path):
    """读取并裁剪图像中心区域"""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape
        cropped_image = image[int(height / 2 - height / 8):int(height / 2 + height / 8),
                            int(width / 2 - width / 8):int(width / 2 + width / 8)]
        return cropped_image
    except Exception as e:
        print(f"Error reading or cropping the image: {e}")
        return None

def histogram_gradient_threshold(image_gray):
    """计算图像直方图及梯度阈值"""
    try:
        hist, bins = np.histogram(image_gray.flatten(), bins=256, range=(0, 256))
        return hist, bins
    except Exception as e:
        print(f"Error calculating histogram: {e}")
        return None, None

def find_peak_hist(hist):
    """寻找直方图的峰值"""
    hmax_index = np.argmax(hist)
    hmax = hist[hmax_index]
    return hmax, hmax_index

def sample_points(hist, peak, bins, step=5):
    """样本点选择"""
    if len(hist) == 0 or len(bins) == 0:
        return None, None
    xmax, ymax = peak
    n = 3  # 采样点数
    x_samples = np.arange(xmax - step * (n - 1), xmax, step)
    y_samples = np.interp(x_samples, bins[:-1], hist)
    return x_samples, y_samples

def fit_line(x, y):
    """直线拟合"""
    if len(x) == 0 or len(y) == 0:
        return None
    p = np.polyfit(x, y, 1)
    if len(p) == 1:  # If p is a scalar (slope is 0)
        return 0
    else:
        return p


def zero_crossing_point(p, bins):
    """计算零交点"""
    if p is None:
        return None
    if len(p) == 1:  # If p is a scalar (slope is 0)
        return float('inf')
    else:
        return -p[1] / p[0] * bins[0]


def plot_histogram_and_threshold(image_gray, threshold):
    """绘制直方图及阈值线"""
    plt.hist(image_gray.flatten(), bins=256, color='gray', alpha=.5, label='Original Image')
    plt.axvline(threshold, 0, linewidth=2, color='r', linestyle='--', label='Threshold')
    plt.title('Histogram with Gradient-based Binarization Threshold')
    plt.legend()
    plt.show()

def plot_binarized_image(image_gray, threshold):
    """显示二值化图像"""
    plt.imshow(image_gray > threshold, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    plt.title('Binarized Image')
    plt.show()

if __name__ == "__main__":
    image_path = '2011.4/11-04-05.992-4.jpg'
    image_gray = read_and_crop_image(image_path)
    if image_gray is not None:
        hist, bins = histogram_gradient_threshold(image_gray)
        if hist is not None and bins is not None:
            hmax, hmax_index = find_peak_hist(hist)
            x_samples, y_samples = sample_points(hist, (hmax, hmax_index), bins, step=5)
            p = fit_line(x_samples, y_samples)
            threshold = zero_crossing_point(p, bins)
            if threshold is not None:
                plot_histogram_and_threshold(image_gray, threshold)
                plot_binarized_image(image_gray, threshold)
            else:
                print("Failed to calculate threshold.")
        else:
            print("Failed to calculate histogram.")
    else:
        print("Failed to read or crop the image.")
