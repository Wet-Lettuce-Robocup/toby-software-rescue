import os

import ultralytics
from ultralytics import YOLO

print(f'Ultralytics version: {ultralytics.__version__}')

# Note: ideally 300+ images of ball, 50-100 of background

model = YOLO('yolov8s.pt')  # n for nano, s for small, anything larger would be too slow.
# .pt for pretrained dataset, .yaml for training with random weights

path = os.path.join(os.getcwd(), 'ml_rescue/ml_training/conf.yaml')
model.train(
    data=path, epochs=100, device='mps', augment=False, imgsz=640, cache='disk', batch=10
)  # Sets training to run on an Apple GPU
