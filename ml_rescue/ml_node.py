import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


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
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # self.get_logger().info('Received image')
            cv2.imshow('Camera View', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
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
