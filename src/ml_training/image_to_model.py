import os
import time

import cv2
from picamera2 import Picamera2
from picamera2.utils import Transform

os.makedirs('raw_images', exist_ok=True)
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        sensor={'output_size': (2304, 1296)},  # 16:9 aspect ratio
        main={'format': 'RGB888', 'size': (1536, 864)},  # Lower resolution for better performance
        controls={'FrameRate': 30},
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

    def start_image_stream(self):
        run = False
        while True:
            frame = picam2.capture_array()
            if frame is not None:
                cv2.imshow('Camera', frame)
                wait = cv2.waitKey(1) & 0xFF  # Wait for 100ms
                if wait == ord(' '):  # ESC key to exit
                    run = True
                if wait == ord('q'):
                    return False

            while run:
                frame = picam2.capture_array()
                if frame is not None:
                    cv2.imshow('Camera', frame)
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

                        cv2.imwrite(image_path, frame)

                        print(f'Captured {image_path}')
                        self.image_count += 1
                else:
                    print('Failed to capture image')
                    print(f'Frame: {frame}')
                time.sleep(0.5)


robot = ImageToModel()
robot.start_image_stream()
