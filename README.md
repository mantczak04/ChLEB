ChLEB: Chessboard Layout Extraction Bot
ChLEB is a computer vision pipeline designed to automatically detect chessboards and recognize chess pieces from images taken at various angles and lighting conditions. The system integrates classical computer vision techniques (Saddle Point detection) with deep learning (YOLOv8) to reconstruct the game state.
🚀 Key Features
* Chessboard Localization: Uses the Hessian matrix to find saddle points (X-corners) for robust board detection.
* Perspective Correction: Automatic perspective warping to generate a normalized $800 \times 800$ px top-down view.
* Piece Detection: Recognition of 12 chess piece classes (Black/White) using YOLOv8.
* Interactive UI: Built with Streamlit to allow real-time adjustment of Canny thresholds, confidence levels, and model selection.
* Dataset Preparation: Automated scripts to crop and warp raw images for model training.
📂 Directory Structure
Plaintext

```
├── data/                   # Test images and datasets
├── src/
│   ├── app.py              # Streamlit Web Interface
│   ├── corner_detection.py # Classical vision logic (Hessian, NMS, Contours)
│   ├── main.py             # CLI testing script
│   ├── prepare_dataset.py  # Preprocessing tool for YOLO training
│   └── utils.py            # Image processing utilities (warping, grayscale)
├── inspect_model.py        # Model metadata inspector
└── requirements.txt        # Project dependencies

```

🛠️ Installation
1. Clone the repository:
Bash

```
git clone https://github.com/your-username/chleb.git
cd chleb

```

2. Install dependencies:
Bash

```
pip install -r requirements.txt

```

3. Add Models:
Place your trained YOLO weights (e.g., `nano_finetuned.pt`) in the root directory.
🖥️ Usage
Interactive Web App
The best way to visualize the detection steps (Saddle points, Canny edges, Contours):
Bash

```
streamlit run src/app.py

```

CLI Batch Processing
Run detection on random samples from the `data/` folder:
Bash

```
python src/main.py

```

🧠 Pipeline Overview
1. Preprocessing: Grayscale conversion, CLAHE (Contrast Limited Adaptive Histogram Equalization), and Gaussian blurring.
2. Saddle Point Detection: Identifying X-junctions using the Hessian determinant to filter out non-chessboard quadrangles.
3. Quadrangle Selection: Ranking contours based on the number of contained saddle points and a "squarishness" metric.
4. Perspective Warp: Mapping the 4 corners to a square grid via a Homography matrix.
5. Inference: Running YOLOv8 on the warped board for high-accuracy piece classification.
