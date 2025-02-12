import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2

# 保持不变的参数和函数定义
blur_ksize = 3


def calculate_histogram(image, range_val=(0, 256)):
    histogram, bin_edges = np.histogram(image.flatten(), bins=256, range=range_val)
    return histogram, bin_edges


def calculate_gradient(histogram):
    gradient = np.diff(histogram)
    # 创建与gradient相同长度的bin_edges
    bin_edges_gradient = np.linspace(0, 255, len(gradient) + 1, endpoint=False)
    return gradient, bin_edges_gradient



def model(x, slope, intercept):
    return slope * x + intercept


def calculate_gradient(histogram):
    gradient = np.diff(histogram)
    # 创建与gradient相同长度的bin_edges
    bin_edges_gradient = np.linspace(0, 255, len(gradient), endpoint=False)  # 将len(gradient) + 1 改为 len(gradient)
    return gradient, bin_edges_gradient


def find_threshold(slope, intercept, bin_edges):
    # 对于过零点的求解，使用更稳健的方法
    zero_crossing = np.argwhere(np.diff(np.sign(slope * bin_edges + intercept))).flatten()
    if len(zero_crossing) > 0:
        threshold = bin_edges[zero_crossing[0]]
    else:
        # 如果没有零点，设置一个默认阈值
        threshold = bin_edges[len(bin_edges) // 2]
    return threshold


def fit_line(gradient, bin_edges_gradient, initial_guess):
    if len(gradient) < 3:
        print("Insufficient data for fitting. Skipping line fitting.")
        return None, None

    # 定义拟合函数
    def fit_func(params, x):
        slope, intercept = params
        residual = slope * x + intercept - gradient
        return residual

    # 进行最小二乘拟合
    result = least_squares(fit_func, initial_guess, args=(bin_edges_gradient,))
    slope, intercept = result.x

    return slope, intercept


def cropped(image):
    height, width = image.shape
    # 确保裁剪后的图像大小合理
    crop_height = min(height // 4, height - height // 4)
    crop_width = min(width // 4, width - width // 4)
    cropped_image = image[int(height / 2 - crop_height / 2):int(height / 2 + crop_height / 2),
                    int(width / 2 - crop_width / 2):int(width / 2 + crop_width / 2)]
    return cropped_image


# 新增异常处理和参数校验

try:
    image_path = '2011.4/11-04-05.992-4.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to load the image.")

    cropped_image = cropped(image)
    img_blur = cv2.GaussianBlur(cropped_image, (blur_ksize, blur_ksize), 1)

    # 增加检查以确保图像内容非空
    if np.all(img_blur == img_blur[0, 0]):
        print("Image content is constant; unable to compute meaningful gradient.")
    else:
        histogram, bin_edges = calculate_histogram(img_blur)
        if np.sum(histogram) == 0:  # 确保直方图中有数据
            print("Histogram is empty; no data for gradient calculation.")
        else:
            gradient, bin_edges_gradient = calculate_gradient(histogram)

            # 检查gradient长度，确保数据充足
            if len(gradient) < 3:
                print("Insufficient data in gradient for fitting.")
            else:
                slope, intercept = fit_line(gradient, bin_edges_gradient, initial_guess=[1, 0])

    # 计算直方图和梯度
    histogram, bin_edges = calculate_histogram(img_blur)
    gradient = calculate_gradient(histogram)

    # 拟合直线
    initial_guess = [1, 0]
    if len(gradient) < 3:
        raise ValueError("Insufficient data for fitting.")

    slope, intercept = fit_line(gradient, bin_edges, initial_guess)

    # 确定二值化阈值
    threshold = find_threshold(slope, intercept, bin_edges)

    # 二值化图像
    bw_image = np.where(img_blur > threshold, 1, 0)

    # 显示直方图和拟合线
    plt.figure()
    plt.hist(img_blur.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.5)
    plt.plot(bin_edges[0:-1], gradient, label='Gradient')
    plt.plot(bin_edges[0:-1], slope * bin_edges[0:-1] + intercept, label='Fitted Line', linestyle='--', color='red')
    plt.axvline(x=threshold, color='green', linestyle='-', label='Threshold')
    plt.legend()
    plt.show()

    # 显示二值化后的图像
    plt.imshow(bw_image, cmap='gray')
    plt.title('Binary Image')
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
