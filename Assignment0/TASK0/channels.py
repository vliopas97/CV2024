#####TUWIEN - WS2024 CV: Task0 - Colorizing Images
#####Liopas+++++++++Evangelos++++12433743
import matplotlib.pyplot as plt
import numpy as np


def corr2d(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate the normalized cross-correlation (NCC) between two input images.
    
    Args:
        img1 (np.ndarray): The first input image (n1 x m1 x 1).
        img2 (np.ndarray): The second input image (n2 x m2 x 1).
    
    Returns:
        float: The normalized cross-correlation coefficient between the images.
    """

    # student_code start

    meanImg1 = np.mean(img1)
    meanImg2 = np.mean(img2)

    corr = np.sum(np.multiply(img1-meanImg1, img2-meanImg2))
    denominator = np.sqrt(
                    np.multiply(np.sum((img1-meanImg1)**2),
                                np.sum((img2-meanImg2)**2)
                        ))
    if(denominator == 0):
        raise RuntimeError("Normalized Cross Correlation error: Division by 0")
    
    corr /= denominator

    #raise NotImplementedError("TO DO in channels.py")

    # student_code end
    
    return corr


def align(imgR: np.ndarray, imgG: np.ndarray, imgB: np.ndarray) -> np.ndarray:
    """
    Align color channels using normalized cross-correlation to create a colorized image.
    HINT: np.roll(..)

    Args:
        imgR (np.ndarray): Image representing the red channel.
        imgG (np.ndarray): Image representing the green channel.
        imgB (np.ndarray): Image representing the blue channel.
    
    Returns:
        np.ndarray: The colorized image (n x m x 3) as a numpy array.
    """
    
    # student_code start
    
    height, width = imgR.shape
    result = np.zeros((height, width, 3))

    if((imgR.shape != imgB.shape) or (imgR.shape != imgG.shape)):
        raise RuntimeError("Dimensions don't match")
    
    # pad the images

    # calculate best displacement for blue and green channel in respect to red
    ncc = {
        'G':float('-inf'),
        'B':float('-inf') 
        }
    
    optimalDisplacement = {
        'G': (0, 0),
        'B': (0, 0)
    }

    for dx in range(-15, 16): # rows
        for dy in range(-15, 16): # columns
            G_shifted = np.roll(imgG, (dx, dy), axis=(0, 1))
            B_shifted = np.roll(imgB, (dx, dy), axis=(0, 1))

            nccG = corr2d(imgR, G_shifted)
            if(nccG > ncc['G']):
                ncc['G'] = nccG
                optimalDisplacement['G'] = (dx, dy)
            
            nccB = corr2d(imgR, B_shifted)
            if(nccB > ncc['B']):
                ncc['B'] = nccB
                optimalDisplacement['B'] = (dx, dy)


    G_ = np.roll(imgG, optimalDisplacement['G'], axis=(0, 1))
    B_ = np.roll(imgB, optimalDisplacement['B'], axis=(0, 1))    

    result = np.dstack((B_, G_, imgR))
    # raise NotImplementedError("TO DO in channels.py")
    # student_code end
                
    return result
