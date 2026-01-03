import cv2
import numpy as np

def order_points(pts):
    """
    Porządkuje punkty w kolejności: 
    [lewy-góra, prawy-góra, prawy-dół, lewy-dół].
    Kluczowe dla poprawnej transformacji perspektywy.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Lewy-góra ma najmniejszą sumę (x+y)
    rect[2] = pts[np.argmax(s)] # Prawy-dół ma największą sumę

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Prawy-góra ma najmniejszą różnicę (y-x)
    rect[3] = pts[np.argmax(diff)] # Lewy-dół ma największą różnicę
    return rect

def corners_detection(image):
    """
    Zamiast szukać wewnętrznych rogów, szuka obrysu planszy (największy czworokąt).
    """
    # 1. Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Rozmycie usuwa szum (np. fakturę drewna), zostawia mocne krawędzie
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 2. Wykrywanie krawędzi (Canny)
    # Parametry 50, 150 są standardowe, ale przy słabym świetle można je zmienić
    edges = cv2.Canny(blur, 50, 150)
    
    # Dylatacja pogrubia krawędzie, żeby zamknąć ewentualne przerwy w obrysie planszy
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # 3. Znajdowanie konturów
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sortujemy kontury od największego (zakładamy, że plansza jest największa)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    board_cnt = None
    
    for cnt in contours:
        # Aproksymacja wielokąta (wygładzanie kształtu)
        peri = cv2.arcLength(cnt, True)
        # 0.02 to współczynnik precyzji - im wyższy, tym bardziej "kanciasty" kształt
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
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
            
        return rect, result_image
    else:
        return None, image