from lifecycle_msgs.srv import ChangeState
import rclpy
from rclpy.node import Node
from sensor_msg.msg import Image


class VisionNode(Node):
    """
    A node.

    - has things
    """

    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('raw_image_topic', '/front_camera/image_raw')

        self.create_subscription(
            Image,
            self.get_parameter('raw_image_topic').get_parameter_value().string_value,
            self.image_callback,
            10,
        )

    def image_callback(self, msg):
        self.get_logger().info('Received image yipee!')

    def change_node_state(self, client, transition_id):
        req = ChangeState.Request()
        req.transition.id = transition_id  # example: Transition.TRANSITION_ACTIVATE
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
