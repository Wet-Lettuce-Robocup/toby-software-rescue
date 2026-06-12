from movement_actions import Movement
from rclpy.node import Node


class MLRescueNode(Node):
    """
    _summary_

    Parameters
    ----------
    Node : _type_
        _description_
    """

    def __init__(self):
        super().__init__('ml_rescue_node')

        robot = Movement()
        robot.drive(1)
