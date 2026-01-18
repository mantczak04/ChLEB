import cv2
import numpy as np
import PIL.Image
import time
import matplotlib.pyplot as plt

def load_image(img_path, out_size=500):
    """
    loads an image and scale it to 500x500 for looking for X-corners
    img_path: string (ścieżka) lub numpy array
    """
    if isinstance(img_path, np.ndarray):
        # Jeśli to już array, konwertuj do PIL
        img_orig = PIL.Image.fromarray(img_path)
    else:
        # Jeśli to ścieżka
        img_orig = PIL.Image.open(img_path)
    
    img = img_orig.resize((out_size, out_size), resample=PIL.Image.BILINEAR)
    return img

def order_points(pts):
    """
    Orders 4 points in the order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    This version is more robust for rotated quads than the sum-difference method.
    """
    # pts shape is (4, 2)
    # 1. Sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # 2. Grab the left-most and right-most points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # 3. Sort left-most by y-coordinate to get Top-Left and Bottom-Left
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # 4. Sort right-most by y-coordinate to get Top-Right and Bottom-Right
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most

    return np.array([tl, tr, br, bl], dtype="float32")

def get_raw_saddle(gray_img):
    # Oczekuje obrazu ~500x500 jako np array
    img = gray_img.astype(np.float64)

    # Blur (3,3) - redukcja szumu przed pochodnymi
    img = cv2.blur(img, (3,3))

    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)
    
    # Hessian: det(H) = gxx*gyy - gxy^2
    # X-corners mają silną negatywną odpowiedź, bierzemy abs dla wizualizacji/score
    S = np.abs(gxx*gyy - gxy**2)
    return S

def nonmax_sup(img, win=10):
    dilated = cv2.dilate(img, np.ones((2*win+1, 2*win+1), np.uint8))

    local_max = (img == dilated)
    img_sup = np.zeros_like(img, dtype=np.float64)
    img_sup[local_max] = img[local_max]

    return img_sup

def prune_saddle(x, threshold=128, score_threshold=10_000):
    score = np.count_nonzero(x > 0)
    current_thresh = threshold
    
    # Pętla zwiększa próg, aż liczba punktów spadnie poniżej limitu
    while score > score_threshold:
        current_thresh *= 2
        x[x < current_thresh] = 0
        score = np.count_nonzero(x > 0)

def get_saddle_points(img):
    # Konwersja PIL -> Numpy grayscale
    gray_img = np.array(img.convert('L'))

    start_time = time.time()
    saddle_img = get_raw_saddle(gray_img)
    raw_saddle = saddle_img.copy()
    
    # Pruning (wstępne czyszczenie słabych punktów)
    prune_saddle(saddle_img)
    
    # Non-Maximum Suppression
    nonmax_saddle_img = nonmax_sup(saddle_img, win=10)

    # Wyciąganie współrzędnych
    pts = np.argwhere(nonmax_saddle_img)

    print(pts.shape, pts.dtype)
    if(len(pts) == 0):
        print('blad')
    
    # POPRAWKA: Było pts[:0] (pusty slice), ma być pts[:,0]
    saddle_scores = nonmax_saddle_img[pts[:,0], pts[:,1]]
    
    ordering = np.argsort(saddle_scores)[::-1] # Sortowanie malejąco
    top_pts = pts[ordering]

    print(f"Znaleziono punktów: {len(top_pts)} (czas: {time.time()-start_time:.3f}s)")
    return top_pts, nonmax_saddle_img, raw_saddle

def run_on_image(img_path):
    # 1. Wczytanie i resize do 500x500
    img = load_image(img_path)

    # 2. Wykrywanie punktów siodłowych
    top_pts, nonmax_saddle_img, raw_saddle = get_saddle_points(img)
    
    # 3. Wizualizacja
    plt.figure(figsize=(15,10))
    
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    # Uwaga: matplotlib używa (x,y), numpy (row, col) -> (y,x)
    plt.plot(top_pts[:,1], top_pts[:,0], 'ro', markersize=2)
    
    # Podpisz 50 najsilniejszych punktów
    label_k = min(50, len(top_pts))
    for i in range(label_k):
        plt.text(top_pts[i,1], top_pts[i,0], str(i), color='yellow', fontsize=8)
    plt.title(f'Found {len(top_pts)} saddle pts')
    
    
    plt.tight_layout()
    plt.show()
    print('Done')

def calculate_squarishness(pts):
    """
    Calculates a score (0 to 1) of how 'squarish' a quadrilateral is.
    Uses ratio of opposite sides and diagonals.
    """
    # pts: [tl, tr, br, bl]
    # Side lengths
    top = np.linalg.norm(pts[0] - pts[1])
    right = np.linalg.norm(pts[1] - pts[2])
    bottom = np.linalg.norm(pts[2] - pts[3])
    left = np.linalg.norm(pts[3] - pts[0])
    
    # Diagonal lengths
    diag1 = np.linalg.norm(pts[0] - pts[2])
    diag2 = np.linalg.norm(pts[1] - pts[3])
    
    # Ratio of opposite sides
    r_width = min(top, bottom) / max(top, bottom)
    r_height = min(left, right) / max(left, right)
    
    # Ratio of diagonals
    r_diag = min(diag1, diag2) / max(diag1, diag2)
    
    # Combine (simple average)
    return (r_width + r_height + r_diag) / 3

def chessboard_edge_detection(image, saddle_points=None):
    """
    looks for the biggest quadrangle that contains saddle points
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
    image = clahe.apply(image)
    #canny edge detector
    edges = cv2.Canny(image, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Increase number of candidates
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    
    best_quad = None
    best_score = -1
    best_quality = 0
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 3000:
            # Order points to calculate metrics
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            
            # 1. Count saddle points
            saddle_count = 0
            if saddle_points is not None:
                for pt in saddle_points:
                    if cv2.pointPolygonTest(approx, (float(pt[1]), float(pt[0])), False) >= 0:
                        saddle_count += 1
            
            # 2. Calculate squarishness
            sq_score = calculate_squarishness(rect)
            
            # Combined score: prioritize saddle points, use squarishness as tie-breaker/refiner
            # We want at least a few saddle points (e.g. > 10)
            if saddle_count < 10:
                continue
                
            # Score formula: (saddle points count) * (squarishness factor)
            # Squarishness factor penalizes highly irregular shapes
            current_score = saddle_count * (sq_score ** 2)
            
            if current_score > best_score:
                best_score = current_score
                best_quad = (rect, approx)
                best_quality = sq_score

    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if best_quad is not None:
        rect, approx = best_quad
        
        # Wizualizacja: Rysujemy zieloną ramkę wokół znalezionej planszy
        cv2.drawContours(result_image, [approx], -1, (0, 255, 0), 3)
        
        # Oznaczamy rogi kolorami, żeby sprawdzić czy kolejność jest OK
        # BGR Order: Blue (TL), Green (TR), Red (BR), Yellow (BL)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] 
        for i, (x, y) in enumerate(rect):
            cv2.circle(result_image, (int(x), int(y)), 15, colors[i], -1)
            
        return rect, result_image, best_quality
    else:
        return None, image, 0
