import cv2
import numpy as np

def to_grayscale_blurred(image) -> np.ndarray:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #change color to grayscale
    image_blur = cv2.GaussianBlur(image_gray, (7,7), 0)       #adding blur
    return image_blur

def warp_chessboard(image, pts, size=800):
    """
    pts: 4 points of the chessboard (ordered)
    size: output size of the square board
    """
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (size, size))
    return warped

def slice_board(warped_image, margin_ratio=0.08): #deprecated
    """
    Slices the warped board into an 8x8 list of square images.
    margin_ratio: how much of the border to skip (0.06 means 6% margin on each side)
    """
    size = warped_image.shape[0]
    margin = int(size * margin_ratio)
    
    # Define the usable area for squares
    board_size = size - 2 * margin
    cell_size = board_size // 8
    
    # Draw grid lines for visualization on a copy
    vis_grid = warped_image.copy()
    for i in range(9):
        # coord of the grid line
        coord = margin + i * cell_size
        cv2.line(vis_grid, (coord, margin), (coord, margin + board_size), (0, 255, 0), 2)
        cv2.line(vis_grid, (margin, coord), (margin + board_size, coord), (0, 255, 0), 2)
    
    squares = []
    for row in range(8):
        row_squares = []
        for col in range(8):
            y1 = margin + row * cell_size
            y2 = y1 + cell_size
            x1 = margin + col * cell_size
            x2 = x1 + cell_size
            
            # Slice with a small vertical buffer (e.g. 20% of cell size) to handle tall pieces
            buffer = int(cell_size * 0.2)
            y1_buffered = max(0, y1 - buffer)
            
            square = warped_image[y1_buffered:y2, x1:x2]
            row_squares.append(square)
        squares.append(row_squares)
            
    return squares, vis_grid
