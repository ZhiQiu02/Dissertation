import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2


blur_ksize = 3


def calculate_histogram(image, range_val=(0, 256)):
    histogram, bin_edges = np.histogram(image.flatten(), bins=256, range=range_val)
    return histogram, bin_edges


def calculate_gradient(histogram):
    gradient = np.diff(histogram)
    return gradient


def model(x, a, b):
    return a * x + b


def fit_line(gradient, bin_edges, initial_guess):
    # 选择靠近梯度拐点的采样点
    sampling_points = np.where((bin_edges > np.argmax(gradient) - 5) & (bin_edges < np.argmax(gradient) + 5))
    sample_edges = bin_edges[sampling_points]
    sample_gradient = gradient[sampling_points]

    # 定义拟合函数
    def fit_func(params, x):
        a, b = params
        residual = a * x + b - sample_gradient
        return residual

    # 进行最小二乘拟合
    result = least_squares(fit_func, initial_guess, args=(sample_edges,))
    a, b = result.x

    return a, b


def find_threshold(a, b):
    # 对于过零点的求解，这里需要一个更稳健的方法
    # 这里只是一个示例，可能需要根据实际情况调整求解方法
    zero_crossing = np.argwhere(np.diff(np.sign(a * bin_edges + b))).flatten()
    threshold = bin_edges[zero_crossing[0]]
    return threshold


def cropped(image):
    height, widths = image.shape
    cropped_image = image[int(height / 2 - height / 8):int(height / 2 + height / 8),
                    int(widths / 2 - widths / 8):int(widths / 2 + widths / 8)]
    return cropped_image


# 假设 image 是加载的灰度图像数据
image_path = '2011.4/11-04-05.992-4.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
cropped_image = cropped(image)
img_blur = cv2.GaussianBlur(cropped_image, (blur_ksize, blur_ksize), 1)


# 计算直方图和梯度
histogram, bin_edges = calculate_histogram(image)
gradient = calculate_gradient(histogram)

# 拟合直线
initial_guess = [1, 0]
a, b = fit_line(gradient, bin_edges, initial_guess)

# 确定二值化阈值
threshold = find_threshold(a, b)

# 二值化图像
bw_image = np.where(image > threshold, 1, 0)

# 显示直方图和拟合线
plt.figure()
plt.hist(image.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.5)
plt.plot(bin_edges[0:-1], gradient, label='Gradient')
plt.plot(bin_edges[0:-1], a * bin_edges[0:-1] + b, label='Fitted Line', linestyle='--', color='red')
plt.axvline(x=threshold, color='green', linestyle='-', label='Threshold')
plt.legend()
plt.show()

# 显示二值化后的图像
plt.imshow(bw_image, cmap='gray')
plt.title('Binary Image')
plt.show()


