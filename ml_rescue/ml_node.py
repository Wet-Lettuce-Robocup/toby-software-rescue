import cv2
from cv_bridge import CvBridge
from hailort import HEF, Device
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray


class VisionNode(Node):
    """
    A node.

    - has things
    """

    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('raw_image_topic', '/front_camera/image_raw')

        self.camera_sub = self.create_subscription(
            Image,
            self.get_parameter('raw_image_topic').get_parameter_value().string_value,
            self.image_callback,
            10,
        )

        self.test_image_sub = self.create_subscription(
            Image,
            '/test_image',
            self.display_callback,
            10,
        )

        self.pub = self.create_publisher(Detection2DArray, '/detections', 10)

        self.bridge = CvBridge()

        self.frame = None
        self.hef_path = 'config/model.hef'  # Path to Hailo model file
        self.hailo = HEF('model.hef')

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame = cv_image

            input_tensor = self.preprocess(cv_image)
            detections = self.hailo_infer(input_tensor)
            detection_msg = Detection2DArray()
            for det in detections:
                detection = Detection2D()
                detection.bbox.center.position.x = det.cx
                detection.bbox.center.position.y = det.cy

                detection.bbox.size_x = det.w
                detection.bbox.size_y = det.h

                detection.results[0].score = det.confidence

                detection_msg.detections.append(detection)

            self.pub.publish(detection_msg)

            # self.get_logger().info('Received image')
            cv2.imshow('Camera View', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            raise

    def preprocess(self, image):
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        return image

    def hailo_infer(self, input_tensor):
        outputs = self.hailo.run(input_tensor)
        detections = self.postprocess(outputs)

        return detections

    def display_callback(self, msg):
        try:
            cv2.imshow('Test Image', self.frame)
            cv2.waitKey(1)
            self.get_logger().info(f'Displayed test image: {msg}')
        except Exception as e:
            self.get_logger().error(f'Error displaying test image: {e}')
            raise


def main(args=None):
    rclpy.init(args=args)
    vision_node = VisionNode()
    rclpy.spin(vision_node)
    vision_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
