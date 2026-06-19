from functools import partial
import os

# import subprocess
import queue

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from hailo_platform import VDevice
import numpy as np
import rclpy
from rclpy.node import Node
from rescue_msgs.srv import EnableInference
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class VisionNode(Node):
    """
    A node.

    - has things
    """

    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('raw_image_topic', '/front_camera/camera_node/image_raw')
        self.declare_parameter('ml_rescue_debug', False)

        self.debug = self.get_parameter('ml_rescue_debug').value

        self.camera_sub = self.create_subscription(
            Image,
            self.get_parameter('raw_image_topic').value,
            self.image_callback,
            10,
        )
        self.test_image_sub = self.create_subscription(
            Image,
            '/ml_rescue/test_image',
            self.test_display_callback,
            10,
        )
        self.inference_pub = self.create_publisher(
            Detection2DArray, '/ml_rescue/inference_stream', 10
        )
        self.rescue_active_srv = self.create_service(
            EnableInference, '/ml_rescue/enable_inference', self.rescue_active_callback
        )
        self.isActive = False

        self.bridge = CvBridge()
        self.image = None

        self.hailo = 'robotyolov8s'  # Model name
        self.hef_path = os.path.join(
            get_package_share_directory('ml_rescue'), 'modelhef', f'{self.hailo}.hef'
        )
        self.imgsz = 640
        self.conf = 0.8
        self.model_classes = ['ball']

        self.results_queue = queue.Queue(maxsize=2)

        self.target = VDevice()
        self.infer_model = self.target.create_infer_model(self.hef_path)
        self.input_name = self.infer_model.input_names[0]
        self.output_name = self.infer_model.output_names[0]
        self.output_shape = self.infer_model.output(self.output_name).shape
        self.configured_model = self.infer_model.configure()

        self.dw = 1536
        self.dh = 864

        self.fps = 10
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            '/videos/output_video.mp4', self.fourcc, self.fps, (self.dw, self.dh)
        )

        self.last_frame = self.get_clock().now()
        self.period = 1 / self.fps

    def rescue_active_callback(self, request, response):
        self.isActive = request.enabled
        response.message = 'Inference enabled' if request.enabled else 'Inference disabled'

        self.get_logger().info(response.message)

        return response

    def image_callback(self, msg):

        if self.isActive:
            now = self.get_clock().now()
            if (now - self.last_frame).nanoseconds < self.period * 1e9:
                return
            self.last_frame = now

            # Convert ROS Image message to OpenCV image
            self.get_logger().info(f'Recieved image: {msg.height}x{msg.width}')
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image = cv_image
            self.out.write(self.image)
            # self.run_inference()

    def run_inference(self):

        raw_frame = self.image
        # self.out.write(raw_frame)
        resized_frame = cv2.resize(raw_frame, (self.imgsz, self.imgsz))
        input_data = np.ascontiguousarray(resized_frame)

        bindings = self.configured_model.create_bindings()
        bindings.input(self.input_name).set_buffer(input_data)

        output_buffer = np.zeros(self.output_shape, dtype=np.float32)
        bindings.output(self.output_name).set_buffer(output_buffer)

        bound_callback = partial(
            self._inference_callback,
            output_buffer=output_buffer,
            display_frame=raw_frame.copy(),
        )

        job = self.configured_model.run_async([bindings], bound_callback)
        self.get_logger().info(job)

        try:
            vis_frame, latest_balls = self.results_queue.get_nowait()

            detection_msg = Detection2DArray()
            detection_msg.header.frame_id = 'cam'

            for ball in latest_balls:
                y1, x1, y2, x2 = ball['box']
                score = ball['score']

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

                detection = Detection2D()
                detection.header.frame_id = 'cam'
                detection.bbox.center.position.x = pxc
                detection.bbox.center.position.y = pyc

                detection.bbox.size_x = px2 - px1
                detection.bbox.size_y = py2 - py1

                detection.results[0].score = score

                detection_msg.detections.append(detection)  # To be sent to ml_rescue_node

                if self.debug:
                    cv2.rectangle(vis_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.circle(vis_frame, (pxc, pyc), 2, (0, 0, 255), -1)
                    self.get_logger().info(
                        f'Object detected at ({pxc},{pyc}) with confidence {score}'
                    )

                    label = f'{self.classes[0]} {score:.2f}'  # Label with confidence score
                    cv2.putText(
                        vis_frame,
                        label,
                        (px1, max(20, py1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
            if latest_balls is None:
                self.get_logger().info('no balls detected')

            # Show the rendered frame on the screen
            if self.debug:
                self.out.write(vis_frame)

        except queue.Empty:
            pass
        if detection_msg is not None:
            self.inference_pub.publish(detection_msg)

    def _inference_callback(self, completion_info, output_buffer=None, display_frame=None):

        flat_buffer = output_buffer.flatten()
        num_detections = int(flat_buffer[0])
        detections = []

        for i in range(num_detections):
            start_idx = 1 + (i * 5)
            y1 = output_buffer[start_idx]
            x1 = output_buffer[start_idx + 1]
            y2 = output_buffer[start_idx + 2]
            x2 = output_buffer[start_idx + 3]
            score = output_buffer[start_idx + 4]

            if score >= self.conf_threshold:
                detections.append({'box': [y1, x1, y2, x2], 'score': score})

        self.get_logger().info(completion_info)

        # Push both the frame and its matching detections to the main thread
        if not self.results_queue.full():
            self.results_queue.put_nowait((display_frame, detections))

    def test_display_callback(self, msg):
        try:
            self.out.write()
            self.get_logger().info(f'Displayed test image: {msg}')
        except Exception as e:
            self.get_logger().error(f'Error displaying test image: {e}')
            raise


def main(args=None):
    rclpy.init(args=args)
    vision_node = VisionNode()
    rclpy.spin(vision_node)
    vision_node.out.release()
    vision_node.destroy_node()
    rclpy.shutdown()
    # command = [
    #     'ffmpeg',
    #     '-y',  # Overwrites the output file if it already exists
    #     '-i',
    #     'videos/output_video.avi',  # Input file
    #     '-vcodec',
    #     'libx264',  # H.264 video codec for compression
    #     '-crf',
    #     '28',  # Compression level (higher = smaller file)
    #     'videos/output.mp4',  # Output file
    # ]
    # subprocess.run(command, check=True)
    # os.remove('videos/output_video.avi')


if __name__ == '__main__':
    main()
