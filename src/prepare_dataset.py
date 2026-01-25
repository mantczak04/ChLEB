import os
import cv2
import numpy as np
from pathlib import Path
import PIL.Image
from tqdm import tqdm

import corner_detection
from utils import to_grayscale_blurred, warp_chessboard

def prepare_yolo_dataset(source_dir, target_dir, output_size=800):
    """
    Takes every image from source_dir, detects the chessboard, 
    warps it, and saves it to target_dir.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Image extensions to process
    img_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [f for f in source_path.iterdir() if f.suffix.lower() in img_extensions]
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    success_count = 0
    fail_count = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        # Load original image
        img_orig = cv2.imread(str(img_file))
        if img_orig is None:
            print(f"Warning: Could not read {img_file.name}")
            fail_count += 1
            continue
            
        orig_h, orig_w = img_orig.shape[:2]
        
        # 1. Prepare for detection (standard 500x500 size used in the project)
        img_500 = cv2.resize(img_orig, (500, 500))
        img_gray = to_grayscale_blurred(img_500)
        
        # 2. Detect saddle points
        # corner_detection.get_saddle_points expects a PIL Image
        pil_img_500 = PIL.Image.fromarray(cv2.cvtColor(img_500, cv2.COLOR_BGR2RGB))
        saddle_points, _, _ = corner_detection.get_saddle_points(pil_img_500)
        
        # 3. Detect chessboard edges
        rect, _, quality = corner_detection.chessboard_edge_detection(img_gray, saddle_points)
        
        if rect is not None:
            # 4. Scale detect points back to original image size
            # rect is a [tl, tr, br, bl] array of (x, y) coordinates in 500x500 space
            scale_x = orig_w / 500.0
            scale_y = orig_h / 500.0
            
            # Create a copy of rect and scale it
            rect_scaled = rect.copy()
            rect_scaled[:, 0] *= scale_x
            rect_scaled[:, 1] *= scale_y
            
            # 5. Warp the original high-res image
            warped = warp_chessboard(img_orig, rect_scaled, size=output_size)
            
            # 6. Save the warped image
            save_path = target_path / img_file.name
            cv2.imwrite(str(save_path), warped)
            success_count += 1
        else:
            print(f"Warning: Chessboard not found in {img_file.name}")
            fail_count += 1
            
    print("\nProcessing complete!")
    print(f"Successfully warped: {success_count}")
    print(f"Failed to detect: {fail_count}")
    print(f"Images saved to: {target_dir}")

if __name__ == "__main__":
    # Define paths
    SOURCE = r"f:/Projekt Widzenie/ChLEB/data/yolo-dataset"
    TARGET = r"f:/Projekt Widzenie/ChLEB/data/yolo-ready"
    
    prepare_yolo_dataset(SOURCE, TARGET)
