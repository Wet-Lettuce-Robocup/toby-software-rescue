from movement_actions import Movement
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray


class MLRescueNode(Node):
    """
    Main node for rescue movement logic.

    Parameters
    ----------
    Node : _type_
        _description_

    """

    def __init__(self):
        super().__init__('ml_rescue_node')

        self.robot = Movement()
        self.robot.drive(1)
        self.vision_sub = self.create_subscription(
            Detection2DArray,
            '/ml_rescue/inference_stream',
            self.inference_callback(),
            10,
        )

    def inference_callback(self, msg):
        self.get_logger().info(f'Recieved: {msg}')
