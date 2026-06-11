from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)
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
            results = self.model(frame)  # Runs inference on video frame
            annotated_image = results[0].plot()  # Displays model-annotated video frame
            cv2.imshow('YOLO', annotated_image)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

    def predict_hailo(self, source):
        # Load the HEF file
        hef = HEF(self.hailo_model_path)

        # Create a VDevice and load the HEF
        params = VDevice().create_params()
        target = VDevice(params)

        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)

        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        # Setup I/O virtual streams
        input_vstreams_params = InputVStreamParams.make(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        output_vstreams_params = OutputVStreamParams.make(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        # Preprocess
        orig = Image.open(source).convert('RGB')
        ow, oh = orig.size
        resized = orig.resize((self.imgsz, self.imgsz))
        input_data = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)  # (1,640,640,3)
        input_name = hef.get_input_vstream_infos()[0].name

        # Inference
        with (
            InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as pipeline,
            network_group.activate(network_group_params),
        ):
            pipeline.send({input_name: input_data})
            raw = pipeline.recv()

        # Parse HailoRT NMS output and draw results
        # When compiled with nms_postprocess the HEF outputs detections grouped by
        # class: shape (batch, num_classes, max_dets, 5) where 5 = [y1,x1,y2,x2,score]
        draw = ImageDraw.Draw(orig)
        output_key = next(iter(raw.keys()))
        batch_dets = raw[output_key][0]  # shape: (num_classes, max_dets, 5)

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
        print('Saved output.jpg')


if __name__ == '__main__':
    pred = PredictionClass()
    # pred.predict_image('test.jpg')
    # pred.predict_pi_video_stream()
    pred.predict_hailo('test.jpg')
