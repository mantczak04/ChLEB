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
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

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

def chessboard_edge_detection(image):
    """
    looks for the biggest quadrangle
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
    image = clahe.apply(clahe)
    #canny edge detector (on preprocessed image)
    edges = cv2.Canny(image, 50, 150)
    
    # Dylatacja pogrubia krawędzie, żeby zamknąć ewentualne przerwy w obrysie planszy
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    #find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sortujemy kontury od największego (zakładamy, że plansza jest największa)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    vis = image.copy()


    board_cnt = None
    
    for cnt in contours:
        # Aproksymacja wielokąta (wygładzanie kształtu)
        peri = cv2.arcLength(cnt, True)
        # 0.02 to współczynnik precyzji - im wyższy, tym bardziej "kanciasty" kształt
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        cv2.drawContours(vis, cnt, -1, 255, 2)
        plt.imshow(vis, cmap='gray')
        plt.show()

        # Szukamy figury, która ma 4 rogi i jest wystarczająco duża
        if len(approx) == 4 and cv2.contourArea(approx) > 2000:
            board_cnt = approx
            break

    result_image = image.copy()
    
    if board_cnt is not None:
        # Konwersja formatu punktów do prostej macierzy (4, 2)
        pts = board_cnt.reshape(4, 2)
        rect = order_points(pts)
        
        # Wizualizacja: Rysujemy zieloną ramkę wokół znalezionej planszy
        cv2.drawContours(result_image, [board_cnt], -1, (0, 255, 0), 3)
        
        # Oznaczamy rogi kolorami, żeby sprawdzić czy kolejność jest OK
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # Niebieski, Zielony, Czerwony, Żółty
        for i, (x, y) in enumerate(rect):
            cv2.circle(result_image, (int(x), int(y)), 15, colors[i], -1)

        # plt.subplot(122)
        # plt.imshow(result_image, cmap='gray')
        # plt.title('NMS Pruned Saddle Map')
            
        return rect, result_image
    else:
        return None, image
