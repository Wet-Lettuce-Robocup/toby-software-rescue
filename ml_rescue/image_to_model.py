import os
import time

import cv2
from picamera2 import Picamera2

os.makedirs('raw_images', exist_ok=True)
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        sensor={'output_size': (2304, 1296)},
        main={'size': (1920, 1080)},
        controls={'FrameRate': 30},
    )
)
picam2.set_controls({'AfMode': 2})
picam2.start()


class ImageToModel:
    """
    A class to capture images using the Raspberry Pi camera
    and save them for training a YOLOv8 model.

    - Captures images from front camera on robot
    - Saves images to 'raw_images' directory
    """

    def __init__(self):
        self.image_count = 0

    def start_image_stream(self):
        while True:
            time.sleep(0.5)
            frame = picam2.capture_array()
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                flipped_image = cv2.flip(frame_bgr, 1)
                cv2.imshow('Camera', flipped_image)
                wait = cv2.waitKey(1)
                if wait % 256 == 27:  # ESC key to exit
                    print('Exiting image capture')
                    picam2.stop()
                    cv2.destroyAllWindows()
                    return False

                elif wait % 256 == 32:  # SPACE key to capture
                    image_path = f'raw_images/image_{self.image_count}.jpg'

                    cv2.imwrite(image_path, flipped_image)

                    print(f'Captured {image_path}')
                    self.image_count += 1
            else:
                print('Failed to capture image')
                print(f'Frame: {frame}')


robot = ImageToModel()
robot.start_image_stream()
