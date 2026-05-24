import os

import ultralytics
from ultralytics import YOLO

print(f'Ultralytics version: {ultralytics.__version__}')

# Note: 300+ images of ball, 50-100 of background

model = YOLO('yolov8s.yaml')  # n for nano, s for small, anything larger would be too slow

path = os.path.join(os.getcwd(), 'ml_rescue/ml_training/conf.yaml')
model.train(data=path, epochs=100, device='mps')  # Sets training to run on an Apple GPU
