#####TUWIEN - WS2024 CV: Task2 - Image Stitching
#####*********+++++++++*******++++GROUP 06
from typing import List, Tuple
from numpy.linalg import inv
import numpy as np
import mapping
import random
import cv2


def get_geometric_transform(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Calculate a homography from the first set of points (p1) to the second (p2)

    Parameters
    ----------
    p1 : np.ndarray
        first set of points
    p2 : np.ndarray
        second set of points
    
    Returns
    ----------
    np.ndarray
        homography from p1 to p2
    """

    num_points = len(p1)
    A = np.zeros((2 * num_points, 9))
    for p in range(num_points):
        first = np.array([p1[p, 0], p1[p, 1], 1])
        A[2 * p] = np.concatenate(([0, 0, 0], -first, p2[p, 1] * first))
        A[2 * p + 1] = np.concatenate((first, [0, 0, 0], -p2[p, 0] * first))
    U, D, V = np.linalg.svd(A)
    H = V[8].reshape(3, 3)

    # homography from p1 to p2
    return (H / H[-1, -1]).astype(np.float32)


def get_transform(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch]) -> Tuple[np.ndarray, List[int]]:
    """
    Estimate the homography between two set of keypoints by implementing the RANSAC algorithm
    HINT: random.sample(..), transforms.get_geometric_transform(..), cv2.perspectiveTransform(..)

    Parameters
    ----------
    kp1 : List[cv2.KeyPoint]
        keypoints left image ([number_of_keypoints] - KeyPoint)
    kp2 :  List[cv2.KeyPoint]
        keypoints right image ([number_of_keypoints] - KeyPoint)
    matches : List[cv2.DMatch]
        indices of matching keypoints ([number_of_matches] - DMatch)
    
    Returns
    ----------
    np.ndarray
        homographies from left (kp1) to right (kp2) image ([3 x 3] - float)
    List[int]
        inliers : list of indices, inliers in 'matches' ([number_of_inliers x 1] - int)
    """

    # student_code start
    N = 1000
    T = 5
    max_inliers = []
    trans = np.zeros((3, 3), dtype=np.float32)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    for _ in range(N):
        # 1: Randomly choose four matches
        selected_indices = random.sample(range(len(matches)), 4)
        src_pts = np.array([pts1[i] for i in selected_indices])
        dst_pts = np.array([pts2[i] for i in selected_indices])

        # 2: Estimate the homography matrix using the four selected points
        H = get_geometric_transform(src_pts, dst_pts)
        if H is None:
            continue
        
        # 3: Transform all points from the first image using the estimated homography
        transformedPoints = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H)

        transformedPoints = transformedPoints.reshape(-1, 2)

        # 4: Calculate the Euclidean distance and count inliers
        distances = np.linalg.norm(pts2 - transformedPoints, axis=1)
        inliers = [i for i, d in enumerate(distances) if d < T]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            trans = H
    # student_code end

    return trans, max_inliers


def to_center(desc: List[np.ndarray], kp: List[cv2.KeyPoint]) -> List[np.ndarray]:
    """
    Prepare all homographies by calculating the transforms from all other images
    to the reference image of the panorama (center image)
    First use mapping.calculate_matches(..) and get_transform(..) to get homographies between
    two consecutive images from left to right, then calculate and return the homographies to the center image
    HINT: inv(..), pay attention to the matrix multiplication order!!
    
    Parameters
    ----------
    desc : List[np.ndarray]
        list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    kp : List[cv2.KeyPoint]
        list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    
    Returns
    ----------
    List[np.ndarray]
        (H_center) list of homographies to the center image ( [number_of_images x 3 x 3] - float)
    """

    # student_code start
    H_center = np.zeros((len(desc), 3, 3))
    h = []

    for i in range(len(desc) - 1):
        matchingKeypoints = mapping.calculate_matches(desc[i], desc[i+1])
        trans, inliers = get_transform(kp[i], kp[i+1], matchingKeypoints)
        h.append(trans)

    # calculate homographies in reference to 3rd (central) image
    H_center[0] = np.matmul(h[1], h[0])
    H_center[1] = h[1]
    H_center[2] = np.identity(3)
    H_center[3] = np.linalg.inv(h[2])
    H_center[4] = np.linalg.inv(np.matmul(h[3], h[2]))
    # student_code end
    
    return H_center


def get_panorama_extents(images: List[np.ndarray], H: List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    """
    Calculate the extent of the panorama by transforming the corners of every image
    and get the minimum and maxima in x and y direction, as you read in the assignment description.
    Together with the panorama dimensions, return a translation matrix 'T' which transfers the
    panorama in a positive coordinate system. Remember that the origin of opencv images is in the upper left corner
    HINT: cv2.perspectiveTransform(..)

    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])

    Returns
    ---------
    np.ndarray
        T : transformation matrix to translate the panorama to positive coordinates ([3 x 3])
    int
        width of panorama (in pixel)
    int
        height of panorama (in pixel)
    """

    # student_code start
    transformed_corners = []
    
    h, w = images[0].shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    # Apply homographies to the corners of each image
    for i in range(len(images)):
        transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H[i]).reshape(-1, 2)
        transformed_corners.append(transformed)
    
    all_corners = np.vstack(transformed_corners)
    
    # Find the min and max x, y coordinates
    min_x, min_y = np.min(all_corners, axis=0)
    max_x, max_y = np.max(all_corners, axis=0)
    
    # Compute the width and height of the panorama
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    
    # Compute the translation matrix to shift the panorama to positive coordinates
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
    
    # student_code end

    return T, width, height
