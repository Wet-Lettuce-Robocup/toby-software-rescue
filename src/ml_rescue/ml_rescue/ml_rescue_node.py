import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_msgs.action import Move
from vision_msgs.msg import Detection2DArray


class Movement:
    """High level movement class that handles robot driving."""

    def __init__(self, node):
        # setup action clients
        self.move_client = ActionClient(node, Move, 'move')

    def drive(self, distance, angle=0, velocity=0.1):
        goal = Move.Goal()

        goal.distance = distance
        goal.angle = angle
        goal.vel = velocity

        self.busy = True

        self.move_client.wait_for_server()

        self.send_goal_future = self.move_client.send_goal_async(goal)

        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:  # if goal is rejected, log error and set busy to false
            self.get_logger().error('Movement Goal rejected')
            self.busy = False
            return

        self.get_logger().info('Movement Goal accepted')

        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def result_callback(self, future):
        result = future.result().result

        if result.success:
            self.get_logger().info('Movement Goal success')
        else:
            self.get_logger().error('Movement Goal fail')

        self.busy = False


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

        self.robot = Movement(self)
        self.robot.drive(1)
        self.vision_sub = self.create_subscription(
            Detection2DArray,
            '/ml_rescue/inference_stream',
            self.inference_callback(),
            10,
        )
        self.dw = 1536
        self.dh = 864

    def inference_callback(self, msg):
        self.get_logger().info(f'Recieved: {msg}')
        self.get_logger().info(f'How does {msg} look and is it within {self.dw} {self.dh}')


def main(args=None):
    rclpy.init(args=args)
    vision_node = MLRescueNode()
    rclpy.spin(vision_node)
    vision_node.destroy_node()
    rclpy.shutdown()
    vision_node.out.release()


if __name__ == '__main__':
    main()
