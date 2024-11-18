#####TUWIEN - WS2024 CV: Task2 - Image Stitching
#####*********+++++++++*******++++GROUP 06
from typing import List, Tuple
from matplotlib.pyplot import angle_spectrum
import numpy as np
# from cyvlfeat.sift.sift import sift
# import cyvlfeat
import os
import cv2
import glob


def get_panorama_data(path: str) -> Tuple[List[np.ndarray], List[cv2.KeyPoint], List[np.ndarray]]:
    """
    Loop through images in given folder (do not forget to sort them), extract SIFT points
    and return images, keypoints and descriptors
    This time we need to work with color images. Since OpenCV uses BGR you need to swap the channels to RGB
    HINT: sorted(..), glob.glob(..), cv2.imread(..), sift=cv2.SIFT_create(), sift.detectAndCompute(..)

    Parameters
    ----------
    path : str
        path to image folder

    Returns
    ---------
    List[np.ndarray]
        img_data : list of images
    List[cv2.Keypoint]
        all_keypoints : list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    List[np.ndarray]
        all_descriptors : list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    """
    # student_code start
    img_data = []
    all_keypoints = []
    all_descriptors = []
    sift = cv2.SIFT.create()

    for filename in os.listdir(path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(path, filename)
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_data.append(image_rgb)

            imgGray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = sift.detectAndCompute(imgGray, None)
            all_keypoints.append(keypoints)
            all_descriptors.append(descriptors)

    return img_data, all_keypoints, all_descriptors
