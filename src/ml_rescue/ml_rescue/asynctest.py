from functools import partial
import queue
import time

import cv2
from hailo_platform import VDevice
from libcamera import Transform
import numpy as np
from picamera2 import Picamera2


class RobotRescueCam:
    """A class for testing asynchronous camera inference."""

    def __init__(self):
        self.hailo_model = 'robotyolov8s'
        self.hailo_model_path = f'config/{self.hailo_model}.hef'
        self.imgsz = 640
        self.conf_threshold = 0.5
        self.classes = ['ball']

        self.picam2 = None

        self.results_queue = queue.Queue(maxsize=2)

    def _inference_callback(self, completion_info, output_buffer=None, display_frame=None):
        """Will be run in the background the exact millisecond the NPU finishes a frame."""
        flat_buffer = output_buffer.flatten()
        num_detections = int(flat_buffer[0])
        detections = []

        for i in range(num_detections):
            start_idx = 1 + (i * 5)
            y1, x1, y2, x2, score = output_buffer[start_idx : start_idx + 5]

            if score >= self.conf_threshold:
                detections.append({'box': [y1, x1, y2, x2], 'score': score})

        print(completion_info)
        # Push both the frame and its matching detections to the main thread
        if not self.results_queue.full():
            self.results_queue.put_nowait((display_frame, detections))

    def run_live_rescue(self):

        if not self.picam2:
            print('Starting Picamera2 hardware...')
            self.picam2 = Picamera2()
            self.picam2.configure(
                self.picam2.create_video_configuration(
                    sensor={'output_size': (2304, 1296)},
                    main={'format': 'BGR888', 'size': (960, 540)},
                    controls={'FrameRate': 30},
                    transform=Transform(hflip=1, vflip=1),
                )
            )
            self.picam2.set_controls({'AfMode': 2})
            self.picam2.start()

        dw = 960
        dh = 540

        # Open the Hailo-10H pipeline
        print('Initialising...')
        with VDevice() as target:
            infer_model = target.create_infer_model(self.hailo_model_path)
            input_name = infer_model.input_names[0]
            output_name = infer_model.output_names[0]
            output_shape = infer_model.output(output_name).shape

            with infer_model.configure() as configured_model:
                print("Rescue System Active - Opening Live Preview... (Press 'q' to quit)")

                try:
                    while True:
                        # Capture the live 960x540 frame from the camera
                        raw_frame = self.picam2.capture_array()

                        # Preprocess: Resize to 640x640 for the model
                        resized_frame = cv2.resize(raw_frame, (self.imgsz, self.imgsz))
                        input_data = np.ascontiguousarray(resized_frame)

                        # Build unique memory buffers for this frame swap
                        bindings = configured_model.create_bindings()
                        bindings.input(input_name).set_buffer(input_data)

                        output_buffer = np.empty(output_shape, dtype=np.float32)
                        bindings.output(output_name).set_buffer(output_buffer)

                        # Throw the frame to the Hailo NPU asynchronously
                        # Pass a copy of raw_frame into the callback package
                        bound_callback = partial(
                            self._inference_callback,
                            output_buffer=output_buffer,
                            display_frame=raw_frame.copy(),
                        )
                        job = configured_model.run_async([bindings], bound_callback)
                        print(job)

                        # Pull finished frames from the queue and render them
                        try:
                            vis_frame, latest_balls = self.results_queue.get_nowait()

                            # Convert colorspace so the screen colors aren't inverted.
                            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

                            for ball in latest_balls:
                                y1, x1, y2, x2 = ball['box']
                                score = ball['score']

                                # Scale fractions directly up to the 960x540 display sizes
                                px1 = int(x1 * dw)
                                py1 = int(y1 * dh)
                                px2 = int(x2 * dw)
                                py2 = int(y2 * dh)

                                # Clamp boxes inside your image frames
                                px1, px2 = max(0, min(dw, px1)), max(0, min(dw, px2))
                                py1, py2 = max(0, min(dh, py1)), max(0, min(dh, py2))

                                cv2.rectangle(vis_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)

                                # Add text label with confidence score
                                label = f'{self.classes[0]} {score:.2f}'
                                cv2.putText(
                                    vis_frame,
                                    label,
                                    (px1, max(20, py1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    2,
                                )

                            # Show the rendered frame on the screen
                            cv2.imshow('Hailo-10H Ball Detection', vis_frame)

                        except queue.Empty:
                            pass

                        # Break out of loop immediately if 'q' is pressed in the windows
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        time.sleep(0.01)

                except KeyboardInterrupt:
                    print('\nShutting down rescue systems...')
                finally:
                    # Clean up all open display windows and camera modules safely
                    cv2.destroyAllWindows()
                    self.picam2.stop()
                    self.picam2.close()
                    print('Camera and windows stopped cleanly.')


if __name__ == '__main__':
    rescue_bot = RobotRescueCam()
    rescue_bot.run_live_rescue()
