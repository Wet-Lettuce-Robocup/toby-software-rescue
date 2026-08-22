from functools import partial
import os
import queue
# import time
# import subprocess


from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
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
from ultralytics import YOLO


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

        self.fps = 20
        self.timer = self.create_timer(1 / self.fps, self.run_inference)

        # Camera and inference setup

        self.isActive = False

        self.bridge = CvBridge()

        self.counts = 0

        self.latest_image = None
        self.latest_cropped_image = None
        self.latest_image_header = None

        self.current_data = {}

        self.model_name = 'state.pt'  # Model filename
        self.model_path = os.path.join(
            get_package_share_directory('ml_rescue'), 'modelpt', f'{self.model_name}'
        )
        self.model = YOLO(self.model_path)
        self.imgsz = 640
        self.conf = 0.6

        self.out = None

        # Frame size
        self.dw = 1536
        self.dh = 864

        self.start_x = 0
        self.start_y = 0

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
        """Enables/disables inference functionality."""
        if request.enabled:
            self.isActive = True
            response.message = 'Inference enabled succesfully'
            self.timer.reset()
        elif not request.enabled:
            self.isActive = False
            response.message = 'Inference disabled'
            self.timer.cancel()
        else:
            self.get_logger().error('Inference toggle failed')

        self.get_logger().info(response.message)
        return response

    def image_callback(self, msg):
        """Stores the latest camera frame."""
        if not self.isActive:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        self.latest_image_header = msg.header
        self.latest_image = frame

        # Cropping for better performance maybe

        height, width = frame.shape[:2]

        bottom_margin = int(height * 0.02)
        start_y = int(height / 3)
        end_y = int(height - bottom_margin)

        side_margin = int(width * 0.05)
        start_x = side_margin
        end_x = width - side_margin

        self.start_x = start_x
        self.start_y = start_y

        self.latest_cropped_image = frame[start_y:end_y, start_x:end_x]

    def crop_box_to_frame(self, x1, y1, x2, y2):
        return (
            int(x1 + self.start_x),
            int(y1 + self.start_y),
            int(x2 + self.start_x),
            int(y2 + self.start_y),
        )

    def run_inference(self):
        if self.latest_cropped_image is None:
            return

        # Creates a copy of latest image so that latest_image can asynchronously update
        image_header = self.latest_image_header
        cropped_frame = self.latest_cropped_image.copy()

        results = self.model.predict(cropped_frame, conf=self.conf, stream=True, imgsz=self.imgsz)

        all_detections = {}

        counts = {'silver': 0, 'black': 0, 'green': 0, 'red': 0}

        for i in results:
            if self.debug:
                annotated_frame = i.plot()
                cv2.imshow('a', annotated_frame)
                cv2.waitKey(1)
                # self.out.write(annotated_frame)

            for x1, y1, x2, y2, conf, cls in i.boxes.data.tolist():
                self.get_logger().info(str(i.boxes))

                x1, y1, x2, y2 = self.crop_box_to_frame(x1, y1, x2, y2)

                class_name = self.model.names[int(cls)]
                self.get_logger().info(f'{cls} detected')

                data = {'cls': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'conf': conf}

                if class_name in counts:
                    counts[class_name] += 1
                all_detections.update({len(all_detections): data})

        all_data = {'header': image_header, 'detections': all_detections, 'counts': counts}

        self.get_logger().info(str(all_data))
        self.current_data = all_data

    def inference_callback(self, request, response):
        """Runs inference on latest camera frame and returns detections."""

        if not self.isActive:
            self.get_logger().warn('Inference called but is not active.')
            response.success = False
            return response

        if self.latest_cropped_image is None:
            self.get_logger().warn('No image received, is the front camera working?')
            response.success = False
            return response

        # Sets a filter for either balls or evacuation points, depending on what is requested
        if request.message == 'ball' or request.message == 'evacpoint':
            pass
        else:
            self.get_logger().info(f'Invalid inference request {request.message}')
            response.success = False
            return response

        detection_msg = None

        all_data = self.current_data

        # Don't parse frames with no detections
        if not all_data['detections']:
            response.success = False
            return response

        image_header = all_data['header']

        # Format data as a Detection2DArray to send to rescue node
        detection_msg = Detection2DArray()
        detection_msg.header.stamp = image_header.stamp
        detection_msg.header.frame_id = image_header.frame_id

        # Filter results based on request
        if request.message == 'ball' and (
            all_data['counts']['silver'] > 0 or all_data['counts']['black'] > 0
        ):
            filter = 'ball'
        elif request.message == 'evacpoint' and (
            all_data['counts']['green'] > 0 or all_data['counts']['red'] > 0
        ):
            filter = 'point'
        else:
            # Don't parse frames with no valid detections
            response.success = False
            return response

        for i in all_data['detections'].values():
            if (filter == 'ball' and i['cls'] == 'silver') or (
                filter == 'point' and i['cls'] == 'green'
            ):
                detection = self.parse_datapoint(i, detection_msg)
                detection_msg.detections.append(detection)  # To be sent to ml_rescue_node

        # Separate for loop to ensure that silver and green have priority
        for i in all_data['detections'].values():
            if (filter == 'ball' and i['cls'] == 'black') or (
                filter == 'point' and i['cls'] == 'red'
            ):
                detection = self.parse_datapoint(i, detection_msg)
                detection_msg.detections.append(detection)  # To be sent to ml_rescue_node

        if detection_msg is not None and detection_msg.detections:
            response.success = True
            response.detections = detection_msg

        else:
            # self.get_logger().info('No detections to send.')
            response.success = False

        return response

    def parse_datapoint(self, datapoint, detection_msg) -> Detection2D:
        detection = Detection2D()
        detection.header = detection_msg.header

        detection.bbox.center.position.x = (datapoint['x2'] + datapoint['x1']) / 2
        detection.bbox.center.position.y = (datapoint['y2'] + datapoint['y1']) / 2
        detection.bbox.size_x = float(datapoint['x2'] - datapoint['x1'])
        detection.bbox.size_y = float(datapoint['y2'] - datapoint['y1'])

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = datapoint['cls']
        hypothesis.hypothesis.score = float(datapoint['conf'])

        detection.results.append(hypothesis)

        return detection


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
            vision_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
