import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


# 高斯滤波核大小
blur_ksize = 9
# Canny 边缘检测高低阈值
canny_lth = 50
canny_hth = 150


def process_basic(img):
    # 基本处理
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blur_gray = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 1)
    # ROI 截取
    # edges = cv2.Canny(img, canny_lth, canny_hth)
    height, widths = gray.shape
    cropped_image = gray[int(height/2 -height/8):int(height/2 + height/8), int(widths/2 - widths/8):int(widths/2 + widths/8)]
    img_blur1 = cv2.GaussianBlur(cropped_image, (blur_ksize, blur_ksize), 1)
    blur_img = cv2.blur(cropped_image, (blur_ksize, blur_ksize))
    median_img = cv2.medianBlur(cropped_image, blur_ksize)
    # cv2.imwrite('blur9.jpg', blur_img)
    # cv2.imwrite('median9.jpg', median_img)
    # cv2.imwrite('gauss9.jpg', img_blur1)
    return img_blur1


def mse(image1, image2):
    # 确保两个图像具有相同的尺寸
    assert image1.shape == image2.shape, "Images must have the same dimensions."
    # 计算均方误差
    mse = np.mean((image1 - image2) ** 2)
    return mse


def calculate_image_snr(original_image, noisy_image):
    signal_power = np.mean(original_image ** 2)

    # 计算噪声功率（带噪声图像与原始图像的差值的平方的均值）
    noise = noisy_image - original_image
    noise_power = np.mean(noise ** 2)

    # 计算信噪比
    snr = 10 * np.log10(signal_power / noise_power)

    return snr


if __name__ == "__main__":

    img = cv2.imread('2011.4/11-04-03.971-4.jpg')
    result = process_basic(img)
    # end = time.time()
    # cv2.imshow('original', result)
    # cv2.waitKey(0)
    # cv2.imwrite('crop.jpg', result)
    # print((end-start))

    block_size = 11
    C = 2
    thresh_adaptive = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    cv2.imwrite('thresh_adaptive1.jpg', thresh_adaptive)
    # 进行直方图均衡化
    equalized_image = cv2.equalizeHist(result)
    cv2.imshow('equalized', equalized_image)
    # cv2.imwrite('equalized1.jpg', equalized_image)
    ret, th = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', th)
    # cv2.imwrite('thresh1.jpg', th)
    ret2, th2 = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('ostu', th2)
    # cv2.imwrite('ostu1.jpg', th2)
    if len(result.shape) == 2:  # 如果图像是灰度图，它只有两个维度（高度和宽度）
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # 转换为BGR图像
    else:  # 如果图像已经是多通道的，则直接使用它
        result_bgr = result

        # 现在你可以将BGR图像转换为HSV图像了
    hsv = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    retinex_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('retinex', retinex_image)
    # cv2.imwrite('retinex.jpg', retinex_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(result)
    cv2.imshow('clahe', clahe_image)
    # cv2.imwrite('clahe.jpg', clahe_image)

    # 2. 计算灰度直方图
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(clahe_image, kernel, iterations=1)

    edges = cv2.Canny(dilated, 100, 200)
    cv2.imshow('edgfe', edges)
    # cv2.imwrite('edges.jpg', edges)
    # 形态学处理 - 膨胀操作，连接断裂的边缘

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = 0
    # 过滤并标记符合条件的轮廓（中心偏析缺陷）
    for contour in contours:
        area = cv2.contourArea(contour)
        if 35 <= area <= 300:  # 面积过滤条件，单位是像素的平方，需要根据实际情况调整
            # 计算轮廓的近似形状
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # 检查轮廓是否是圆形或椭圆形（通过检查顶点的数量）
            if len(approx) >= 5:  # 圆形或椭圆形通常有较多顶点，这里设为5或以上
                num_contours += 1
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                cv2.circle(result, (cX, cY), 5, (0, 0, 255), -1)  # 红色圆点标记中心

                # 绘制轮廓（可选，用于可视化）
                cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)  # 绿色轮廓线
    cv2.imshow('final', result)
    cv2.waitKey(0)
    '''

    hist = cv2.calcHist([thresh_adaptive], [0], None, [256], [0, 255])

    # 3. 绘制灰度直方图

    plt.rcParams['font.family'] = 'SimSun'  # 或其他已安装的支持中文的字体名

    plt.figure(figsize=(10, 6))
    plt.hist(thresh_adaptive.ravel(), bins=256, range=[0, 256], edgecolor='black')
    plt.xlabel('图像灰度级')
    plt.ylabel('像素个数')

    # 使用 tick_params() 方法旋转y轴标签
    plt.tick_params(axis='y', labelrotation=90)

    plt.xticks(range(0, 257, 25))
    plt.grid(axis='y', alpha=0.5)
    plt.show()
'''

    '''
    # Sobel边缘检测（计算x方向梯度）
    sobel_x = cv2.Sobel(th, cv2.CV_64F, 1, 0, ksize=3)

    # Sobel边缘检测（计算y方向梯度）
    sobel_y = cv2.Sobel(th, cv2.CV_64F, 0, 1, ksize=3)

    # 结合x和y方向梯度，计算梯度幅值和方向
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_direction = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))

    # 将幅值和方向转换为8位整型
    gradient_magnitude_uint8 = cv2.convertScaleAbs(gradient_magnitude)
    gradient_direction_uint8 = cv2.convertScaleAbs(gradient_direction * 180 / np.pi)

    # 为显示方便，将梯度方向转换为0-180度范围
    gradient_direction_uint8[gradient_direction_uint8 > 180] = 180

    # 显示Sobel边缘检测结果
    cv2.imshow("Gradient Magnitude", gradient_magnitude_uint8)
    cv2.imshow("Gradient Direction", gradient_direction_uint8)
    edges = cv2.Canny(th, 100, 200)
    cv2.imshow("Gradient edge", edges)

    laplacian = cv2.Laplacian(th, cv2.CV_64F)
    laplacian_uint8 = cv2.convertScaleAbs(laplacian)
    cv2.imshow("Laplacian Edge Detection", laplacian_uint8)
    cv2.waitKey(0)
    
    '''
# 2011.4/11-04-05.992-4.jpg