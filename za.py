import cv2
import numpy as np
import os
import glob
blur_ksize = 3


def remove_small_noise(img, output_path=None, kernel_size=3):
    # 创建方形结构元素（可调整大小）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # 进行腐蚀操作
    eroded_img = cv2.erode(img, kernel)
    return eroded_img


def roberts_edge_detection(img, output_path=None):
    # Robert算子核
    roberts_kernel_x = np.array([[1, 0], [0, -1]])
    roberts_kernel_y = np.array([[0, 1], [-1, 0]])

    # 计算x和y方向梯度
    grad_x = cv2.filter2D(img, -1, roberts_kernel_x)
    grad_y = cv2.filter2D(img, -1, roberts_kernel_y)

    # 合并梯度
    roberts_edges = np.sqrt(grad_x**2 + grad_y**2)
    # 转换为8位整型
    roberts_edges_uint8 = cv2.convertScaleAbs(roberts_edges.astype(np.uint8))
    return roberts_edges_uint8


def sobel_edge_detection(img, output_path=None):
    # Sobel算子核
    sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # 计算x和y方向梯度
    grad_x = cv2.filter2D(img, cv2.CV_64F, sobel_kernel_x)
    grad_y = cv2.filter2D(img, cv2.CV_64F, sobel_kernel_y)

    # 合并梯度
    sobel_edges = np.sqrt(grad_x**2 + grad_y**2)

    # 转换为8位整型
    sobel_edges_uint8 = cv2.convertScaleAbs(sobel_edges.astype(np.uint8))
    return sobel_edges_uint8


def canny_edge_detection(img, output_path=None):
    # Canny算子参数（可按需调整）
    low_threshold = 50
    high_threshold = 150

    # 进行Canny边缘检测
    canny_edges = cv2.Canny(img, low_threshold, high_threshold)
    return canny_edges


def remove_illumination(image):
    # 使用高斯滤波估计光照背景
    blur = cv2.GaussianBlur(image, (21, 21), 0)
    # 通过原图减去背景来去除光照不均
    return cv2.subtract(image, blur)


def threshold_segmentation(image):
    # 应用阈值分割
    threshold = 0.384
    _, binary_image = cv2.threshold(image, threshold * 255, 255, cv2.THRESH_BINARY)
    return binary_image


def cropped(image):
    height, widths = image.shape
    cropped_image = image[int(height / 2 - height / 8):int(height / 2 + height / 8),
                    int(widths / 2 - widths / 8):int(widths / 2 + widths / 8)]
    return cropped_image


def retinex(input_image):
    scales = [30, 80, 220]
    log_images = np.zeros((input_image.shape[0], input_image.shape[1], len(scales)))

    # 计算图像的对数域
    for i, scale in enumerate(scales):
        log_images[:, :, i] = np.log(1 + input_image.astype(float) / scale)

    # 计算图像的反射率
    log_reflectance = np.mean(log_images, axis=2)
    reflectance = np.exp(log_reflectance)

    # 调整范围并转换为uint8类型
    reflectance = np.uint8(255 * (reflectance - np.min(reflectance)) / (np.max(reflectance) - np.min(reflectance)))

    return reflectance


def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否读取成功
    if image is None:
        print("Error: Could not read image.")
        return None

    cropped_image1 = cropped(image)
    # cv2.imwrite('crop2.jpg', cropped_image)
    img_blur = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 1)
    enhanced = cv2.equalizeHist(img_blur)
    reflectance = retinex(enhanced)
    thresholded = threshold_segmentation(reflectance)
    cropped_image2 = cropped(thresholded)
    # cv2.imwrite('thre111.jpg', cropped_image2)
    # cv2.imshow('thre', cropped_image2)
    eroded = remove_small_noise(cropped_image2)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    edges3 = canny_edge_detection(dilated)
    # cv2.imshow('canny', edges3)
    # cv2.imwrite('canny1.jpg', edges3)
    # 寻找轮廓
    contours, _ = cv2.findContours(edges3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤并标记符合条件的轮廓（中心偏析缺陷）
    for contour in contours:
        cv2.drawContours(cropped_image1, [contour], -1, (0, 0, 255), 2)  # 绿色轮廓线
        area = cv2.contourArea(contour)
        '''
        if 10 <= area <= 300:  # 面积过滤条件，单位是像素的平方，需要根据实际情况调整
                # 绘制轮廓（可选，用于可视化）
                cv2.drawContours(cropped_image1, [contour], -1, (0, 0, 255), 2)  # 轮廓线
'''
    return cropped_image1


input_dir = '/2011.4'
output_dir = 'final'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def process_images(input_dir, output_dir):
    # 验证input_dir是否为安全路径，避免路径遍历攻击
    # 此处的实现只是一个示例，具体的验证逻辑取决于应用环境
    if not os.path.isabs(input_dir) or '..' in input_dir.split(os.sep):
        print("Invalid directory path.")
        return

    try:
        # 规范化路径，避免因路径结尾的'/'等原因导致的意外结果
        input_dir = os.path.normpath(input_dir)

        # 使用集合来管理支持的图片格式，提高可维护性
        supported_formats = {'jpg', 'jpeg', 'png', 'gif'}
        pattern = os.path.join(input_dir, '*.')
        image_files = []

        for format in supported_formats:
            image_files.extend(sorted(glob.glob(pattern + format.upper() + '.*')))
            image_files.extend(sorted(glob.glob(pattern + format.lower() + '.*')))

        # 对找到的文件进行处理
        for image_path in image_files:
            # 使用with语句确保文件正确关闭
            with open(image_path, 'rb') as image_file:  # 示例：打开文件
                # 进行图片处理
                # 请根据实际情况替换下面的代码
                image = cv2.imread(image_path)

                # 应用图片处理，例如增加对比度
                processed_image = preprocess_image(image)

                # 分离文件名和扩展名
                base_name, ext = os.path.splitext(os.path.basename(image_path))

                # 构建输出文件名，例如 "processed_" + 原始文件名 + .ext
                output_file = os.path.join(output_dir, f'processed_{base_name}{ext}')

                # 使用imwrite保存处理后的图片
                cv2.imwrite(output_file, processed_image)

    except FileNotFoundError:
        print(f"Directory {input_dir} not found.")
    except PermissionError:
        print(f"Permission denied for directory {input_dir}.")


# 示例用法
process_images(input_dir, output_dir)

'''
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. 计算灰度直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 255])

# 3. 绘制灰度直方图


plt.rcParams['font.family'] = 'SimSun'  # 或其他已安装的支持中文的字体名

plt.figure(figsize=(10, 6))
plt.hist(img.ravel(), bins=256, range=[0, 256], edgecolor='black')
plt.xlabel('图像灰度级')
plt.ylabel('像素个数')

# 使用 tick_params() 方法旋转y轴标签
plt.tick_params(axis='y', labelrotation=90)

plt.xticks(range(0, 257, 25))
plt.grid(axis='y', alpha=0.5)
plt.show()
'''
