import cv2
import numpy as np

def to_grayscale_blurred(image) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #change color to grayscale
    image = cv2.GaussianBlur(image, (7,7), 0)       #adding blur
    cv2.namedWindow('chessboard', cv2.WINDOW_NORMAL)
    cv2.imshow('chessboard', image)
    cv2.waitKey(0)
    return image