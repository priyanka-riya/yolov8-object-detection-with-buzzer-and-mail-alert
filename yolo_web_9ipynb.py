# -*- coding: utf-8 -*-
"""yolo_web.9ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RKxTh_jzpmrDieRu976UsgBSr-T_FHlL
"""

!pip install ultralytics
!pip install opencv-python
!pip install numpy
!pip install torch
!pip install torch

from ultralytics import YOLO
import cv2
import time
import torch

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Object class names (COCO dataset)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize FPS counter
prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image")
        break

    # Get FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Run YOLO detection
    results = model(img)

    for r in results:
        for box in r.boxes:
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Confidence score
            confidence = round(float(box.conf[0]), 2)

            # Class index
            cls = int(box.cls[0].item())
            class_name = classNames[cls]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Display label and confidence
            label = f"{class_name} {confidence}"
            cv2.putText(img, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display FPS
    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show output
    cv2.imshow('YOLOv8 Object Detection', img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()