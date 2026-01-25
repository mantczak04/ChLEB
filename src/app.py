import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import corner_detection
from utils import to_grayscale_blurred, warp_chessboard, slice_board
import os

# Set page config
st.set_page_config(page_title="ChLEB - Chessboard Detection", layout="wide")

st.title("ChLEB: Chessboard & Piece Detection")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Model selection
available_models = [f for f in os.listdir("..") if f.endswith(".pt")]
# Fallback if none found in root (though we saw some in list_dir)
if not available_models:
    available_models = ["medium_finetuned.pt", "nano_finetuned.pt", "medium_no_finetune.pt"]

model_option = st.sidebar.selectbox(
    "Select Model",
    available_models,
    index=0 if "nano_finetuned.pt" in available_models else 0
)

# Confidence threshold
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.65, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Chessboard Detection Params")
canny_low = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
canny_high = st.sidebar.slider("Canny High Threshold", 0, 255, 150)
approx_epsilon = st.sidebar.slider("Approx Poly Epsilon", 0.01, 0.10, 0.03, 0.005)
min_area_ratio = st.sidebar.slider("Min Board Area (% of image)", 0.0, 1.0, 0.01, 0.01)
min_saddle = st.sidebar.number_input("Min Saddle Points", value=6, step=1)

# Additional output settings
st.sidebar.markdown("---")
show_steps = st.sidebar.checkbox("Show Processing Steps")

# Image uploader
uploaded_file = st.file_uploader("Choose a chessboard image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image / Board Detection")
        
        # Process image for detection
        img_500 = cv2.resize(img, (500, 500))
        img_processed = to_grayscale_blurred(img_500)
        
        # 1. Detect X-corners
        pil_img = Image.fromarray(cv2.cvtColor(img_500, cv2.COLOR_BGR2RGB))
        saddle_points, _, _ = corner_detection.get_saddle_points(pil_img)

        # 2. Detect chessboard edges
        rect, board_img, quality, edges_img, contours = corner_detection.chessboard_edge_detection(
            img_processed, 
            saddle_points,
            canny_thresholds=(canny_low, canny_high),
            approx_epsilon=approx_epsilon,
            min_area_ratio=min_area_ratio,
            min_saddle=min_saddle
        )
        
        if rect is not None:
            st.image(cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB), caption=f"Detected Board (Quality: {quality:.2f})")
        else:
            st.warning("Could not find chessboard in the image.")
            st.image(cv2.cvtColor(img_500, cv2.COLOR_BGR2RGB), caption="Original Image")

    if show_steps:
        st.markdown("---")
        st.subheader("Processing Steps")
        step_col1, step_col2, step_col3 = st.columns(3)
        
        with step_col1:
            # Draw saddle points
            saddle_vis = img_500.copy()
            for pt in saddle_points:
                cv2.circle(saddle_vis, (int(pt[1]), int(pt[0])), 3, (0, 0, 255), -1)
            st.image(cv2.cvtColor(saddle_vis, cv2.COLOR_BGR2RGB), caption=f"Saddle Points ({len(saddle_points)} found)")
            
        with step_col2:
            st.image(edges_img, caption="Dilated Edges (Canny + Morph)")
            
        with step_col3:
            # Draw top contours
            contour_vis = img_500.copy()
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
            st.image(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB), caption=f"Top {len(contours)} Contours")

    with col2:
        st.subheader("Piece Detection")
        
        if rect is not None:
            # 3. Load YOLO model
            @st.cache_resource
            def load_yolo_model(model_path):
                return YOLO(os.path.join("..", model_path))
            
            try:
                model = load_yolo_model(model_option)
                
                # 4. Warp perspective
                # Use the original image for high-res warping, but rect is in 500x500
                orig_h, orig_w = img.shape[:2]
                scale_x = orig_w / 500.0
                scale_y = orig_h / 500.0
                
                rect_scaled = rect.copy()
                rect_scaled[:, 0] *= scale_x
                rect_scaled[:, 1] *= scale_y
                
                warped = warp_chessboard(img, rect_scaled)
                
                # 5. Run piece detection
                results = model.predict(warped, conf=conf_threshold, verbose=False)
                
                # 6. Plot results
                res_plotted = results[0].plot()
                st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption=f"Detected Pieces ({len(results[0])} found)")
                
                # Optional: Show grid if user wants
                if st.checkbox("Show 8x8 Grid"):
                    _, board_with_grid = slice_board(warped)
                    st.image(cv2.cvtColor(board_with_grid, cv2.COLOR_BGR2RGB), caption="Warped Board with Grid")
                    
            except Exception as e:
                st.error(f"Error running model: {e}")
        else:
            st.info("Detect a chessboard first to run piece detection.")

else:
    st.info("Please upload an image to start.")
    
    # Show some info/instructions
    st.markdown("""
    ### How it works:
    1. **Upload** an image of a chessboard.
    2. The app **resizes** it and searches for saddle points (X-corners).
    3. It finds the **best fitting quadrangle** (the board).
    4. The board is **warped** to a square perspective.
    5. **YOLO** detects chess pieces on the warped board.
    """)
