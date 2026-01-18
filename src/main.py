import os
from pathlib import Path
import cv2
import corner_detection
from utils import to_grayscale_blurred, warp_chessboard, slice_board
import PIL.Image
import numpy as np
import random

import matplotlib.pyplot as plt

def main():
    data_library = Path("data")

    image_extensions = [".jpg", ".png"]

    images = [f for f in data_library.iterdir() if f.suffix.lower() in image_extensions]

    random_samples = 10
    random_images = random.sample(images, random_samples)
    for img_path in random_images:
        print(f"working on {img_path.name}...")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"can't load {img_path.name}")
            continue
        
        img = cv2.resize(img, (500, 500))
        img_processed = to_grayscale_blurred(img)
        
        # 1. Detect X-corners
        pil_img = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        saddle_points, _, _ = corner_detection.get_saddle_points(pil_img)

        # 2. Detect chessboard edges using saddle points
        rect, board_img, quality = corner_detection.chessboard_edge_detection(img_processed, saddle_points)
        
        if rect is not None:
            # 3. Warp perspective
            warped = warp_chessboard(img, rect)
            
            # 4. Slice board into 8x8 grid
            squares, board_with_grid = slice_board(warped)
            
            # 5. Show results using matplotlib
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Board Det: {img_path.name} (Q: {quality:.2f})")
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(cv2.cvtColor(board_with_grid, cv2.COLOR_BGR2RGB))
            plt.title("Warped Board with Grid")
            plt.axis('off')

            # Show a few sample squares (e.g. from the first row) in the third subplot
            plt.subplot(133)
            # Create a small montage of the first 4 squares
            sample_squares = squares[0][:4]
            montage = np.hstack([cv2.resize(s, (100, 100)) for s in sample_squares])
            plt.imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
            plt.title("Sample Squares (0:0-0:3)")
            plt.axis('off')

            plt.show()
        else:
            print(f"could not find chessboard in {img_path.name}")

if __name__ == "__main__":
    main()