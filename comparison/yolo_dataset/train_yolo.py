#!/usr/bin/env python3
"""
# YOLO Training Script for Custom Shelf Analysis Model
"""
from ultralytics import YOLO

# Load a base model
model = YOLO('yolov8n.pt')  # Start with YOLOv8 nano

# Train the model on our custom dataset
results = model.train(
    data='dataset.yaml',  # Use relative path since we're in yolo_dataset directory
    epochs=100,
    imgsz=640,
    batch=8,  # Reduced batch size for CPU
    name='shelf_analysis_custom',
    patience=20,
    save=True,
    device='cpu'  # Use CPU instead of auto
)

print("Training completed!")
print(f"Results saved in: runs/detect/shelf_analysis_custom/")
