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
        main={'format': 'RGB888', 'size': (1536, 864)},  # Lower resolution for better performance
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
        run = False
        while True:
            frame = picam2.capture_array()
            if frame is not None:
                undistorted_frame = cv2.undistort(
                    frame,
                    self.camera_matrix,
                    self.distortion_coefficients,
                    None,
                    self.camera_matrix,
                )
                cv2.imshow('Camera', undistorted_frame)
                wait = cv2.waitKey(1) & 0xFF  # Wait for 100ms
                if wait == ord(' '):  # ESC key to exit
                    run = True
                if wait == ord('q'):
                    return False

            while run:
                start_time = time.time()
                frame = picam2.capture_array()
                if frame is not None:
                    undistorted_frame = cv2.undistort(
                        frame,
                        self.camera_matrix,
                        self.distortion_coefficients,
                        None,
                        self.camera_matrix,
                    )
                    cv2.imshow('Camera', undistorted_frame)
                    wait = cv2.waitKey(1) & 0xFF  # Wait for 100ms
                    if wait == ord('q'):  # 'q' to exit
                        print('Exiting image capture')
                        picam2.stop()
                        cv2.destroyAllWindows()
                        return False
                    elif wait == ord(' '):
                        run = False
                    else:
                        image_path = f'raw_images/image_{self.image_count}.jpg'

                        cv2.imwrite(image_path, undistorted_frame)

                        print(f'Captured {image_path}')
                        self.image_count += 1
                else:
                    print('Failed to capture image')
                    print(f'Frame: {frame}')
                elapsed_time = time.time() - start_time
                if elapsed_time < SPF:
                    time.sleep(SPF - elapsed_time)


robot = ImageToModel()
robot.start_image_stream()
