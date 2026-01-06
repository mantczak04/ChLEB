import os
from pathlib import Path
import cv2
import corner_detection

def main():
    data_library = Path("data")

    image_extensions = [".jpg", ".png"]

    images = [f for f in data_library.iterdir() if f.suffix.lower() in image_extensions]
    i=0
    for img_path in images:
        i=i+1
        if i==3:
            break
        print(f"working on {img_path.name}...")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"can't load {img_path.name}")
        
        corner_detection.run_on_image(image)
        # corners, result = corners_detection(image)
        # if corners is not None:
        #     cv2.namedWindow("rogi", cv2.WINDOW_NORMAL)
        #     cv2.imshow("rogi", result)
        #     cv2.waitKey(0)

        # else:
        #     print("no corners found")
        

if __name__ == "__main__":
    main()