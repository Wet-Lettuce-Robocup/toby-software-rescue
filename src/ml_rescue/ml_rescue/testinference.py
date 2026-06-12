from hailo_platform import VDevice
from hailo_platform.pyhailort.pyhailort import InferModel
import numpy as np
from PIL import Image, ImageDraw
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
        self.pt_model = YOLO('config/best.pt')  # Loads custom model from training
        self.picam2 = None  # Placeholder

        self.hailo_model = 'robotyolov8s'
        self.hailo_model_path = f'config/{self.hailo_model}.hef'  # Loads custom model from training
        self.imgsz = 640
        self.conf_threshold = 0.25
        self.classes = ['ball']

    def predict_image(self, image_path):
        results = self.pt_model(image_path)  # Runs inference on test image
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
            results = self.pt_model(frame)  # Runs inference on video frame
            annotated_image = results[0].plot()  # Displays model-annotated video frame
            cv2.imshow('YOLO', annotated_image)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

    def predict_hailo(self, source):
        # 1. Preprocess image BEFORE entering the hardware context
        orig = Image.open(source).convert('RGB')
        ow, oh = orig.size
        resized = orig.resize((self.imgsz, self.imgsz))

        # Hailo-10H InferModel expects contiguous raw uint8 (640, 640, 3), NOT float32, NO batch dimension
        input_data = np.ascontiguousarray(np.array(resized, dtype=np.uint8))

        # 2. Open the modern Hailo-10H VDevice pipeline
        print('Initializing Hailo-10H VDevice...')
        with VDevice() as target:
            print('Loading model via the InferModel API...')
            infer_model = target.create_infer_model(self.hailo_model_path)

            # Fetch the precise model layer names
            input_name = infer_model.input_names[0]
            output_name = infer_model.output_names[0]

            # 3. Configure the chip and run inference
            with infer_model.configure() as configured_model:
                print('Creating bindings...')
                bindings = (
                    configured_model.create_bindings()
                )  # Pre-allocates buffers based on model input/output shapes

                bindings.input(input_name).set_buffer(input_data)  # Binds uint8 input frame

                output_shape = infer_model.output(output_name).shape
                output_buffer = np.empty(output_shape, dtype=np.float32)
                bindings.output(output_name).set_buffer(output_buffer)

                print('Running inference...')
                # 3. Pass the unified bindings object into run (and include the timeout ms)
                configured_model.run([bindings], 1000)
                print('Inference successful!')

        # 4. Parse HailoRT NMS output and draw results
        draw = ImageDraw.Draw(orig)

        # Extract the raw matrix for the batch frame
        batch_dets = output_buffer[0]  # shape: (num_classes, max_dets, 5)

        for cls_idx, cls_dets in enumerate(batch_dets):
            for det in cls_dets:
                score = float(det[4])
                if score < self.conf_threshold:
                    continue
                y1, x1, y2, x2 = det[:4]

                # Scale from model coords (0-640) back to original image size
                x1 = int(x1 * ow / self.imgsz)
                y1 = int(y1 * oh / self.imgsz)
                x2 = int(x2 * ow / self.imgsz)
                y2 = int(y2 * oh / self.imgsz)

                label = f'{self.classes[cls_idx]} {score:.2f}'
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                draw.text((x1 + 2, y1 + 2), label, fill='red')

        orig.save('output.jpg')
        print('Saved output.jpg successfully!')

    def test_hailo(self):

        hef_path = 'config/robotyolov8s.hef'

        print('Initializing Hailo-10H VDevice...')
        with VDevice() as target:
            print('Loading model via the InferModel API...')
            infer_model = target.create_infer_model(hef_path)

            input_names = infer_model.input_names
            output_names = infer_model.output_names

            with infer_model.configure() as configured_model:
                print('Success! The Hailo-10H accepted the model using InferModel.')

                print('\nModel Bindings:')
                print(f'Inputs:  {input_names}')
                print(f'Outputs: {output_names}')

                # 1. Use .shape property to look inside the stream
                for name in input_names:
                    input_stream = infer_model.input(name)
                    shape = input_stream.shape  # Returns something like (640, 640, 3)

                    # 2. Calculate the exact frame size in bytes (Width * Height * Channels)
                    frame_size_bytes = np.prod(shape)

                    print(f" -> Input '{name}' Dimensions: {shape}")
                    print(f" -> Input '{name}' Expected Buffer Size: {frame_size_bytes} bytes")


if __name__ == '__main__':
    pred = PredictionClass()
    # pred.predict_image('test.jpg')
    # pred.predict_pi_video_stream()
    pred.predict_hailo('test.jpg')
