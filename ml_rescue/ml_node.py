import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VisionNode(Node):
    """
    A node.

    - has things
    """

    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('raw_image_topic', '/front_camera/image_raw')

        self.create_subscription(
            String,
            self.get_parameter('raw_image_topic').get_parameter_value().string_value,
            self.image_callback,
            10,
        )
