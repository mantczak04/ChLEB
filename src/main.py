import os
from pathlib import Path
import cv2
import corner_detection
from utils import to_grayscale_blurred, warp_chessboard, slice_board
import PIL.Image
import numpy as np
import random

import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = 'nano_finetuned.pt'
CONF_THRESHOLD = 0.65

#yolo classes to fen equivalent
CLASS_TO_FEN = {
    'white-pawn': 'P', 'white-rook': 'R', 'white-knight': 'N', 'white-bishop': 'B', 'white-queen': 'Q', 'white-king': 'K',
    'black-pawn': 'p', 'black-rook': 'r', 'black-knight': 'n', 'black-bishop': 'b', 'black-queen': 'q', 'black-king': 'k'
}

def main():
    # Load model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    data_library = Path("data")

    image_extensions = [".jpg", ".png"]

    images = [f for f in data_library.iterdir() if f.suffix.lower() in image_extensions]

    random_samples = 5
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
            
            #board slice #deparcated
            #squares, board_with_grid = slice_board(warped)
            
            # 5. Run piece detection
            results = model.predict(warped, conf=CONF_THRESHOLD, verbose=False)
            print(f"Detected {len(results[0])} pieces in {img_path.name}")
            
            # 6. Show results using matplotlib
            plt.figure(figsize=(15, 5))
            
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Board Det: {img_path.name}")
            plt.axis('off')

            plt.subplot(122)
            # Use YOLO's build-in visualization for detections
            res_plotted = results[0].plot()
            plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
            plt.title("Detected Pieces")
            plt.axis('off')

            plt.show()
        else:
            print(f"could not find chessboard in {img_path.name}")

if __name__ == "__main__":
    main()