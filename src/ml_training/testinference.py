import ultralytics
from ultralytics import YOLO

print(f'Ultralytics version: {ultralytics.__version__}')


class PredictionClass:
    """
    A class for making predictions using a YOLO model.

    - Loads a custom-trained YOLO model
    - Predicts on test images or a video stream from Raspberry Pi camera
    """

    def __init__(self):
        self.model = YOLO('config/best.pt')  # Loads custom model from training
        self.picam2 = None  # Placeholder

    def predict_image(self, image_path):
        results = self.model(image_path)  # Runs inference on test image
        results[0].show()  # Displays model-annotated image

    def predict_pi_video_stream(self, frames=100):
        if not self.picam2:
            import cv2
            from libcamera import Transform
            from picamera2 import Picamera2

            self.picam2 = Picamera2()
            self.picam2.configure(
                self.picam2.create_video_configuration(
                    sensor={'output_size': (2304, 1296)},  # Max is 4608x2592
                    main={'format': 'RGB888', 'size': (960, 540)},
                    controls={'FrameRate': 10},
                    transform=Transform(hflip=1, vflip=1),  # 180 degree rotation
                )
            )
            self.picam2.set_controls({'AfMode': 2})
            self.picam2.start()

        for i in range(frames):
            frame = self.picam2.capture_array()
            results = self.model(frame)  # Runs inference on video frame
            annotated_image = results[0].plot()  # Displays model-annotated video frame
            cv2.imshow('YOLO', annotated_image)
            cv2.waitKey(1)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    pred = PredictionClass()
    # pred.predict_image('test.jpg')
    pred.predict_pi_video_stream()
