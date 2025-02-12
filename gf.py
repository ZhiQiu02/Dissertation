import cv2
import numpy as np
import matplotlib.pyplot as plt
blur_ksize = 5


# 读取原始图像
imagepath = '2011.4/11-04-05.992-4.jpg'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)


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

gray = read_and_crop_image(imagepath)
# 使用自适应直方图均衡化增强对比度
img_blur = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 1)
enhanced = cv2.equalizeHist(img_blur)


# 定义并应用Retinex算法去除反光
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


reflectance = retinex(enhanced)

# 二值化
threshold = 0.384
_, binary_image = cv2.threshold(reflectance, threshold * 255, 255, cv2.THRESH_BINARY)
height, width = binary_image.shape
cropped_image = binary_image[int(height / 2 - height / 8):int(height / 2 + height / 8),
                    int(width / 2 - width / 8):int(width / 2 + width / 8)]
# img_blur = cv2.GaussianBlur(cropped_image, (blur_ksize, blur_ksize), 1)
# 显示图像
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(reflectance, cmap='gray')
plt.title('De-Glared Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cropped_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

# 显示直方图
plt.subplot(2, 2, 4)
plt.hist(reflectance.ravel(), bins=64)
plt.ylabel('Number of Pixels')
plt.xlabel('Pixel Intensity')
plt.title('Histogram')

plt.tight_layout()
plt.show()