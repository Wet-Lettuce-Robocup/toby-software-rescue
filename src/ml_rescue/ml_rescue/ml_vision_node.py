from functools import partial
import os
import queue
# import time
# import subprocess


from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from hailo_platform import VDevice
import numpy as np
import rclpy
from rclpy.node import Node
from rescue_msgs.srv import EnableInference, InferenceDetections
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class VisionNode(Node):
    """Runs ML model on video stream and returns inference data to rescue_detections service."""

    def __init__(self):
        super().__init__('vision_node')

        # Initialising ROS topics/services

        self.declare_parameter('raw_image_topic', '/front_camera/camera_node/image_raw')
        self.declare_parameter('ml_rescue_debug', True)

        self.debug = self.get_parameter('ml_rescue_debug').value
        self.get_logger().info(f'Vision node debug status: {self.debug}')

        self.camera_sub = self.create_subscription(
            Image,
            self.get_parameter('raw_image_topic').value,
            self.image_callback,
            1,
        )
        self.rescue_detections_srv = self.create_service(
            InferenceDetections, 'rescue_detections', self.inference_callback
        )
        self.rescue_active_srv = self.create_service(
            EnableInference, 'enable_inference', self.rescue_active_callback
        )

        # Camera and inference setup

        self.isActive = False

        self.bridge = CvBridge()

        self.fps = 10

        self.latest_image = None
        self.latest_image_header = None

        self.hailo = 'robotyolov8s'  # Model filename
        self.hef_path = os.path.join(
            get_package_share_directory('ml_rescue'), 'modelhef', f'{self.hailo}.hef'
        )
        self.imgsz = 640
        self.conf_threshold = 0.7
        self.model_classes = ['black', 'silver']

        self.results_queue = queue.Queue(maxsize=2)

        try:
            self.target = VDevice()
            self.infer_model = self.target.create_infer_model(self.hef_path)
            self.input_name = self.infer_model.input_names[0]
            self.output_name = self.infer_model.output_names[0]
            self.output_shape = self.infer_model.output(self.output_name).shape
            self.configured_model = self.infer_model.configure()

        except Exception as e:
            self.get_logger().error(f'Error loading hailort: {e}')
            self.target = None
            raise

        self.out = None

        # Frame size
        self.dw = 1536
        self.dh = 864

        # Video writer setup for debugging
        pipeline = (
            'appsrc ! queue '
            f'! video/x-raw,format=BGR,width={self.dw},height={self.dh},framerate={self.fps}/1 '
            '! videoconvert '
            '! x264enc bitrate=8000 tune=zerolatency speed-preset=fast '
            '! mp4mux fragment-duration=1000 ! filesink location=/videos/output_video.mp4'
        )
        self.out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, self.fps, (self.dw, self.dh))
        self.get_logger().info(f'Video writer opened: {self.out.isOpened()}')

    def rescue_active_callback(self, request, response):
        """Gets request to toggle inference (enabled/disabled)."""
        self.isActive = request.enabled
        response.message = (
            'Inference enabled successfully' if request.enabled else 'Inference disabled'
        )

        self.get_logger().info(response.message)

        return response

    def image_callback(self, msg):
        """Updates self.latest_image with the latest camera frame."""
        if not self.isActive:
            return

        self.latest_image_header = msg.header
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def inference_callback(self, request, response):
        """Gets request for either balls or evacuation points, and returns detections."""
        # self.get_logger().info('Inference has been called.')

        if not self.isActive:
            self.get_logger().warn('Inference called but is not active.')
            response.success = False
            return response

        if self.latest_image is None:
            self.get_logger().warn('No image received, is the front camera working?')
            response.success = False
            return response

        # Sets a filter for either balls or evacuation points, depending on what is requested
        if request.message == 'ball':
            mode = 1
        elif request.message == 'evacpoint':
            mode = 2
        else:
            self.get_logger().info(f'Invalid inference request {request.message}')
            response.success = False
            return response

        # start_time = time.time()

        # Creates a copy of latest image so that latest_image can asynchronously update and not break things
        image_header = self.latest_image_header
        raw_frame = self.latest_image.copy()
        detection_msg = None

        # self.out.write(raw_frame)

        # Convert frame to an array for processing
        resized_frame = cv2.resize(raw_frame, (self.imgsz, self.imgsz))
        input_data = np.ascontiguousarray(resized_frame)

        # Create Hailo bindings
        bindings = self.configured_model.create_bindings()
        bindings.input(self.input_name).set_buffer(input_data)

        output_buffer = np.zeros(self.output_shape, dtype=np.float32)
        bindings.output(self.output_name).set_buffer(output_buffer)

        bound_callback = partial(
            self._inference_callback,
            output_buffer=output_buffer,
            display_frame=raw_frame,
        )

        # self.configured_model.wait_for_async_ready(timeout_ms=1000)

        # if self.inferenceBusy:
        #     return

        # self.inferenceBusy = True

        # Run inference and wait for return
        job = self.configured_model.run_async([bindings], bound_callback)
        job.wait(1000)

        try:
            vis_frame, latest_balls = self.results_queue.get_nowait()

            # self.get_logger().info(f'balls: {latest_balls}')

            # Format data as a Detection2DArray to send to rescue node
            detection_msg = Detection2DArray()
            detection_msg.header.stamp = image_header.stamp
            detection_msg.header.frame_id = image_header.frame_id

            for ball in latest_balls:
                self.get_logger().info('Ball detected')
                y1, x1, y2, x2 = ball['box']
                score = ball['score']
                material = ball['class_name']

                # Scale values back to original frame size
                px1 = int(x1 * self.dw)
                py1 = int(y1 * self.dh)
                px2 = int(x2 * self.dw)
                py2 = int(y2 * self.dh)

                # Clamp boxes inside your image frames
                px1, px2 = max(0, min(self.dw, px1)), max(0, min(self.dw, px2))
                py1, py2 = max(0, min(self.dh, py1)), max(0, min(self.dh, py2))

                pxc = (px1 + px2) / 2
                pyc = (py1 + py2) / 2

                # # Backup code if colour detection breaks
                # roi = vis_frame[py1:py2, px1:px2]
                # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # mean = gray.mean()

                # if mean > 100:
                #     material = 'silver'
                #     # self.get_logger().info(f'Mean is {mean} detected as silver')
                # else:
                #     material = 'black'
                #     # self.get_logger().info(f'Mean is {mean} detected as black')

                # Filter results based on request
                if (
                    mode == 1
                    and material in ['silver', 'black']
                    or mode == 2
                    and material in ['green', 'red']
                    or material == 'obstacle'
                ):
                    detection = Detection2D()
                    detection.header = detection_msg.header

                    detection.bbox.center.position.x = pxc
                    detection.bbox.center.position.y = pyc
                    detection.bbox.size_x = float(px2 - px1)
                    detection.bbox.size_y = float(py2 - py1)

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = material
                    hypothesis.hypothesis.score = float(score)

                    detection.results.append(hypothesis)
                    detection_msg.detections.append(detection)  # To be sent to ml_rescue_node

                if self.debug:
                    # Output an annotated frame with objects outlined
                    pxc, pyc, px1, py1, px2, py2 = map(int, (pxc, pyc, px1, py1, px2, py2))

                    # Red outline for bounding box and dot for centre of object
                    cv2.rectangle(vis_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.circle(vis_frame, (pxc, pyc), 2, (0, 0, 255), -1)
                    # self.get_logger().info(
                    #     f'Object detected at ({pxc},{pyc}) with confidence {score}'
                    # )

                    label = f'{material} {score:.2f}'  # Label with confidence score
                    cv2.putText(
                        vis_frame,
                        label,
                        (px1, max(20, py1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

            # inf_time_elapsed = time.time() - start_time
            # if self.debug:
            #     self.get_logger().info(f'--- Just for inference: {inf_time_elapsed:.2f} ---')

            # Render frame to video
            if self.debug:
                self.out.write(vis_frame)

        except queue.Empty:
            pass

        if detection_msg is not None and detection_msg.detections:
            # self.get_logger().info('- - - Publishing detections - - -')
            response.success = True
            response.detections = detection_msg

        else:
            # self.get_logger().info('No detections to send.')
            response.success = False

        return response

        # time_elapsed = time.time() - start_time
        # if self.debug:
        #     self.get_logger().info(f'--- Total time elapsed: {time_elapsed:.2f}s ---')
        #     self.get_logger().info(
        #         f'Time for contouring: {(time_elapsed - inf_time_elapsed):.2f}s'
        #     )

    def _inference_callback(self, completion_info, output_buffer=None, display_frame=None):
        """Callback that runs post-processing to clean up data."""

        flat_buffer = output_buffer.flatten()
        detections = []

        idx = 0
        vals = 5  # Each object has 5 points of data
        max_dets = 100  # Maximum of 100 detections are returned, should lower in future

        # self.get_logger().info(f'output_buffer.shape = {output_buffer.shape}')
        # self.get_logger().info(f'flat_buffer[:80] = {flat_buffer[:80]}')

        for class_id, class_name in enumerate(self.model_classes):
            if idx >= len(flat_buffer):
                self.get_logger().warn(
                    f'Ran out of buffer before reading count for class {class_name} at idx={idx}'
                )
                break
            num_detections = int(flat_buffer[idx])
            idx += 1

            # self.get_logger().info(f'class {class_name}: {num_detections} detections')

            num_detections = max(0, min(num_detections, max_dets))  # just in case

            # Iterate through detections
            for det_i in range(num_detections):
                if idx + vals > len(flat_buffer):
                    self.get_logger().warn(
                        f'Ran out of buffer while reading detection {det_i} for class {class_name}'
                    )
                    break

                y1 = flat_buffer[idx]
                x1 = flat_buffer[idx + 1]
                y2 = flat_buffer[idx + 2]
                x2 = flat_buffer[idx + 3]
                score = flat_buffer[idx + 4]

                idx += vals

                # Only return data points with a high enough confidence
                if score >= self.conf_threshold:
                    detections.append({
                        'box': [y1, x1, y2, x2],
                        'score': score,
                        'class_id': class_id,
                        'class_name': class_name,
                    })

        # Push both the frame and its matching detections to the main thread
        if not self.results_queue.full():
            self.results_queue.put_nowait((display_frame, detections))

        # self.inferenceBusy = False


def main(args=None):
    try:
        rclpy.init(args=args)
        vision_node = VisionNode()
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    finally:
        if vision_node is not None:
            if vision_node.out is not None:
                vision_node.out.release()
            if vision_node.target is not None:
                vision_node.target.release()
            vision_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
