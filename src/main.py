import os
from pathlib import Path
import cv2
import corner_detection
from utils import to_grayscale_blurred
import PIL.Image
import numpy as np

def main():
    data_library = Path("data")

    image_extensions = [".jpg", ".png"]

    images = [f for f in data_library.iterdir() if f.suffix.lower() in image_extensions]

    for img_path in images[:2]:
        print(f"working on {img_path.name}...")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"can't load {img_path.name}")
            continue
        
        img = cv2.resize(img, (500, 500))
        img_processed = to_grayscale_blurred(img)
        corners = corner_detection.run_on_image(img_processed)

        result = corner_detection.chessboard_edge_detection(img_processed)
        
        

if __name__ == "__main__":
    main()