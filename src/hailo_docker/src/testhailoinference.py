from hailo_platform import VDevice
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
        self.picam2 = None  # Placeholder

        self.hailo_model = 'robotyolov8s'
        self.hailo_model_path = f'models/{self.hailo_model}.hef'
        self.imgsz = 640
        self.conf_threshold = 0.25
        self.classes = ['black', 'silver']
        self.timeout_ms = 1000

    def old_predict_hailo(self, source):

        orig = Image.open(source).convert('RGB')
        ow, oh = orig.size
        resized = orig.resize((self.imgsz, self.imgsz))

        input_data = np.ascontiguousarray(np.array(resized, dtype=np.uint8))

        print('Initialising Hailo-10H VDevice...')
        with VDevice() as target:
            print('Loading model...')
            infer_model = target.create_infer_model(self.hailo_model_path)

            input_name = infer_model.input_names[0]
            output_name = infer_model.output_names[0]

            # Configure the chip and run inference
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
                configured_model.run([bindings], self.timeout_ms)
                print('Inference successful!')

        # Parses HailoRT NMS flat output array from the assigned output buffer
        draw = ImageDraw.Draw(orig)

        # Extract how many valid detections were found (Index 0)
        num_detections = int(output_buffer[0])
        print(f'Parsing detections. Valid objects found: {num_detections}')

        # Loop through only the valid boxes using 5-step strides
        for i in range(num_detections):
            start_idx = 1 + (i * 5)

            y1 = output_buffer[start_idx]
            x1 = output_buffer[start_idx + 1]
            y2 = output_buffer[start_idx + 2]
            x2 = output_buffer[start_idx + 3]
            score = output_buffer[start_idx + 4]

            print(f'Raw coords: {y1, x1, y2, x2}')
            print(f'Score: {score}')

            if score < self.conf_threshold:
                print(f'Score {score} is below threshold {self.conf_threshold}')
                continue

            # Multiply fractions directly by the original image dimensions
            x1_scaled = int(x1 * ow)
            y1_scaled = int(y1 * oh)
            x2_scaled = int(x2 * ow)
            y2_scaled = int(y2 * oh)

            # Bound boxes to keep them inside image borders
            x1_scaled = max(0, min(ow, x1_scaled))
            y1_scaled = max(0, min(oh, y1_scaled))
            x2_scaled = max(0, min(ow, x2_scaled))
            y2_scaled = max(0, min(oh, y2_scaled))

            label = f'{self.classes[0]} {score:.2f}'

            # Draw the bounding box onto the image
            draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], outline='red', width=3)
            draw.text((x1_scaled + 5, y1_scaled + 5), label, fill='red')

        orig.save('output.jpg')
        print('Saved output.jpg successfully!')

    def old_test_hailo(self):

        hef_path = self.hailo_model_path

        print('Initialising...')

        with VDevice() as target:
            print('Loading model...')
            infer_model = target.create_infer_model(hef_path)

            input_names = infer_model.input_names
            output_names = infer_model.output_names

            with infer_model.configure() as configured_model:
                print(configured_model.__doc__)  # Idk man
                print('Success!')

                print('\nModel Bindings:')
                print(f'Inputs:  {input_names}')
                print(f'Outputs: {output_names}')

                for name in input_names:
                    input_stream = infer_model.input(name)
                    shape = input_stream.shape

                    # Calculate the exact frame size in bytes (Width * Height * Channels)
                    frame_size_bytes = np.prod(shape)

                    print(f" -> Input '{name}' Dimensions: {shape}")
                    print(f" -> Input '{name}' Expected Buffer Size: {frame_size_bytes} bytes")

    def predict_hailo_async(self):
        with VDevice() as target:
            print('VDevice loaded.')

            infer_model = target.create_infer_model(self.hailo_model_path)
            input_name = infer_model.input_names[0]
            output_name = infer_model.output_names[0]
            output_shape = self.infer_model.output(self.output_name).shape

            with infer_model.configure() as configured_model:
                print('Model configured')

                bindings = configured_model.create_bindings()

                buffer = np.zeros(infer_model.input().shape, dtype=np.uint8)
                bindings.input().set_buffer(buffer)

                buffer = np.zeros(infer_model.output().shape, dtype=np.uint8)
                bindings.output().set_buffer(buffer)

                # Run inference synchronously initially
                configured_model.run([bindings], self.timeout_ms)

                # Holds results after inference completes, without blocking NPU
                buffer = bindings.output().get_buffer()
                print(f'Synchronous inference done - output shape: {buffer.shape}')

                print('Starting asynchronous inference...')
                job = configured_model.run_async([bindings])
                job.wait(self.timeout_ms)

                # Wait for the inference to complete and get results
                job.result()  # This will block until inference is done
                print('Inference completed! Output buffer is ready.')


if __name__ == '__main__':
    pred = PredictionClass()
    # pred.predict_image('test.jpg')
    # pred.predict_pi_video_stream()
    pred.predict_hailo_async()
