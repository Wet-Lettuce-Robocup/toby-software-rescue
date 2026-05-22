import os
import time

import cv2

os.makedirs('raw_images', exist_ok=True)
cap = cv2.VideoCapture(0)


class ImageToModel:
    """
    A class to capture images using the Raspberry Pi camera and save them for training a YOLOv8 model.

    - Captures images from front camera on robot
    - Saves images to 'raw_images' directory
    """

    def __init__(self):
        self.image_count = 0

    def start_image_stream(self):
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Camera', frame)
                wait = cv2.waitKey(1)
                if wait % 256 == 27:  # ESC key to exit
                    print('Exiting image capture')
                    cap.release()
                    cv2.destroyAllWindows()
                    return False

                elif wait % 256 == 32:  # SPACE key to capture
                    image_path = f'raw_images/image_{self.image_count}.jpg'
                    image = cv2.imread(image_path)
                    flipped_image = cv2.flip(image, 1)
                    cv2.imwrite(image_path, flipped_image)

                    print(f'Captured {image_path}')
                    self.image_count += 1
            else:
                print('Failed to capture image')


robot = ImageToModel()
robot.start_image_stream()
