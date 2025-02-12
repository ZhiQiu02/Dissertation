import cv2
import numpy as np
from skimage import exposure, filters, morphology
from sklearn.cluster import KMeans


def load_and_preprocess_image(img_path):
    # 1. 加载图片并转换为灰度图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 2. 光照校正
    # （a）直方图均衡化
    img_eq = exposure.equalize_hist(img)
    img_eq = img_eq.astype(np.uint8)

    # （b）CLAHE (对比受限自适应直方图均衡化)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # 3. 高斯滤波降噪
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

    return img_eq, img_clahe, img_blur


def apply_thresholding(img, method, **kwargs):
    assert img.dtype == np.uint8 and img.ndim == 2, "Input image should be a single-channel uint8 grayscale image."
    if method == "global":
        # 全局阈值法
        _, thresh = cv2.threshold(img, kwargs.get("threshold", 127), 255, cv2.THRESH_BINARY_INV)
    elif method == "adaptive_mean":
        # 自适应阈值法（均值）
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                       kwargs.get("block_size", 51), kwargs.get("C", 8))
    elif method == "adaptive_gaussian":
        # 自适应阈值法（高斯加权）
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                       kwargs.get("block_size", 51), kwargs.get("C", 8))
    elif method == "otsu":
        # Otsu's二值化
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "kmeans":
        # K-means聚类阈值分割
        kmeans = KMeans(n_clusters=2).fit(np.reshape(img, (-1, 1)))
        labels = kmeans.labels_
        thresh = np.where(labels == 0, 0, 255).reshape(img.shape)
    else:
        raise ValueError(f"Unsupported thresholding method: {method}")

    return thresh

def main(img_path):
    img_eq, img_clahe, img_blur = load_and_preprocess_image(img_path)
    assert img_blur.dtype == np.uint8 and img_blur.ndim == 2, "Image should be a single-channel uint8 grayscale image."
    # 尝试多种阈值分割方法
    methods = {
        "global": {"threshold": 190},
        "adaptive_mean": {"block_size": 11, "C": 10},
        "adaptive_gaussian": {"block_size": 51, "C": 10},
        "otsu": {},
        # "kmeans": {}
    }
    assert img_blur.dtype == np.uint8 and img_blur.ndim == 2, "Image should be a single-channel uint8 grayscale image."
    for i, (method, params) in enumerate(methods.items()):
        thresh = apply_thresholding(img_blur, method, **params)

        # 为了便于对比，对分割结果进行形态学开运算（去除小噪声）
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        cv2.imshow(f"{method} - Preprocessed", img_blur)
        cv2.imshow(f"{method} - Thresholding", opening)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = "2011.4/11-04-05.992-4.jpg"
    main(img_path)

