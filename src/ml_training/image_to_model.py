import os
import time
import yaml

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.utils import Transform

FPS = 2
SPF = 1 / FPS

os.makedirs('raw_images', exist_ok=True)
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        sensor={'output_size': (2304, 1296)},  # 16:9 aspect ratio
        main={
            'format': 'RGB888',
            'size': (384, 216),
        },  # Even lower resolution for better performance
        controls={'FrameRate': 10},
        transform=Transform(hflip=True, vflip=True),  # 180 degree rotation
    )
)
picam2.set_controls({'AfMode': 2})
picam2.start()


class ImageToModel:
    """
    Capture images using the Raspberry Pi camera and saves them for training a YOLOv8 model.

    - Captures images from front camera on robot
    - Saves images to 'raw_images' directory
    """

    def __init__(self):
        self.image_count = 0

        with open('ost.yaml', 'r') as f:
            calib_data = yaml.safe_load(f)
        raw_matrix = calib_data['camera_matrix']

        raw_dist = calib_data['distortion_coefficients']

        self.camera_matrix = np.array(raw_matrix['data'], dtype=np.float32).reshape(
            raw_matrix['rows'], raw_matrix['cols']
        )
        self.distortion_coefficients = np.array(raw_dist['data'], dtype=np.float32).reshape(
            raw_dist['rows'], raw_dist['cols']
        )

    def start_image_stream(self):
        running = True
        capturing = False
        while running:
            start_time = time.time()

            cropped_frame = self.capture_frame(picam2.capture_array())

            wait = cv2.waitKey(1) & 0xFF  # Wait for 100ms
            if wait == ord(' '):  # ESC key to exit
                capturing = not capturing
            if wait == ord('q'):
                running = False

            if capturing:
                image_path = f'raw_images/image_{self.image_count}.jpg'

                cv2.imwrite(image_path, cropped_frame)

                print(f'Captured {image_path}')
                self.image_count += 1

            elapsed_time = time.time() - start_time
            if elapsed_time < SPF:
                time.sleep(SPF - elapsed_time)

    def capture_frame(self, frame):
        if frame is not None:
            undistorted_frame = cv2.undistort(
                frame,
                self.camera_matrix,
                self.distortion_coefficients,
                None,
                self.camera_matrix,
            )

            cropped_frame, top_left, bottom_right = self.crop_frame(undistorted_frame)
            display_frame = undistorted_frame.copy()
            cv2.rectangle(display_frame, top_left, bottom_right, (0, 0, 255), 2)
            cv2.imshow('Frame', display_frame)

            return cropped_frame

    def crop_frame(self, frame):
        height, width = frame.shape[:2]

        bottom_margin = int(height * 0.02)
        start_y = int(height / 3)
        end_y = int(height - bottom_margin)

        side_margin = int(width * 0.05)
        start_x = side_margin
        end_x = width - side_margin

        return (frame[start_y:end_y, start_x:end_x]), (start_x, start_y), (end_x - 1, end_y - 1)


robot = ImageToModel()
robot.start_image_stream()
picam2.stop()
cv2.destroyAllWindows()
