from ultralytics import YOLO
import sys

try:
    model = YOLO("best.pt")
    print(f"Model task: {model.task}")
    print(f"Model names: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
