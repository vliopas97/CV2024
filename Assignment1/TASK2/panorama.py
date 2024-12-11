#####TUWIEN - WS2024 CV: Task2 - Image Stitching
#####*********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_simple(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    """
    Stitch the final panorama with the calculated panorama extents
    by transforming every image to the same coordinate system as the center image. Use the dot product
    of the translation matrix 'T' and the homography per image 'H' as transformation matrix.
    HINT: cv2.warpPerspective(..), cv2.addWeighted(..)
    
    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    width : int
        width of panorama (in pixel)
    height : int
        height of panorama (in pixel)
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])
    T : np.ndarray
        translation matrix for panorama ([3 x 3])

    Returns
    ---------
    np.ndarray
        (result) panorama image ([height x width x 3])
    """
    
    # student_code start
    
    result = np.zeros((height, width, 3), dtype=np.uint32)

    for i, image in enumerate(images):
        transform = np.matmul(T, H[i])
        imageTransformed = cv2.warpPerspective(image, transform, (width, height))
        result += imageTransformed

    # student_code end
    return result


def get_blended(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    """
    Use the equation from the assignment description to overlay transformed
    images by blending the overlapping colors with the respective alpha values
    HINT: ndimage.distance_transform_edt(..)
    
    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    width : int
        width of panorama (in pixel)
    height : int
        height of panorama (in pixel)
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])
    T : np.ndarray
        translation matrix for panorama ([3 x 3])

    Returns
    ---------
    np.ndarray
        (result) blended panorama image ([height x width x 3])
    """
    
    # student_code start
    result = np.zeros((height, width, 3), dtype=np.float64)
    weights = np.zeros((height, width), dtype = np.float64)

    for i, image in enumerate(images):
        alpha = np.zeros(image.shape[:2], dtype=np.float64)
        borderMask = np.ones_like(alpha)
        borderMask[:1, :] = 0
        borderMask[-1: , :] = 0
        borderMask[:, :1] = 0
        borderMask[:, -1:] = 0
        distances = ndimage.distance_transform_edt(borderMask)
        alpha = distances / distances.max()

        transform = np.matmul(T, H[i])
        imageTransformed = cv2.warpPerspective(image, transform, (width, height))
        alphaTransformed = cv2.warpPerspective(alpha, transform, (width, height))
        
        # Sigma_{i=0}^n I_c * a_i
        for channel in range(3):
            result[:, :, channel] += imageTransformed[:, :, channel] * alphaTransformed

        weights += alphaTransformed

    
    weights[weights == 0] = 1  # To avoid division by zero
    for channel in range(3):
        result[:, :, channel] = result[:, :, channel]/weights # divide by # Sigma_{j=0}^n a_j

    result = np.clip(result, 0, 255).astype(np.uint8)

    # student_code end

    return result
