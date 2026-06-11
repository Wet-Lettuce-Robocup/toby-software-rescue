from movement_actions import Movement
from rclpy.node import Node


class MLRescueNode(Node):
    def __init__(self):
        super().__init__('ml_rescue_node')
