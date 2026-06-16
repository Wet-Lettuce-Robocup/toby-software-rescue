import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_msgs.action import Move
from vision_msgs.msg import Detection2DArray


class Movement:
    def __init__(self, node):
        self.node = node
        # setup action clients
        self.move_client = ActionClient(node, Move, 'move')
        # runtime state
        self.busy = False
        self.current_angle = 0.0
        self.distance_travelled = 0.0
        self.angle_turned = 0.0

    def drive(self, distance, angle=0, velocity=0.1):
        goal = Move.Goal()

        goal.distance = distance
        goal.angle = angle
        goal.vel = velocity

        self.busy = True

        # wait for action server to appear (short timeout so it fails fast if it's not available)
        try:
            available = self.move_client.wait_for_server(timeout_sec=2.0)
        except Exception as e:
            self.node.get_logger().error(f'wait_for_server exception: {e}')
            self.busy = False
            return

        if not available:
            self.node.get_logger().error('Action server "move" not available (timeout)')
            self.busy = False
            return

        try:
            # register feedback callback so we get ongoing updates
            self.send_goal_future = self.move_client.send_goal_async(
                goal, feedback_callback=self.feedback_callback
            )
            self.send_goal_future.add_done_callback(self.goal_response_callback)
        except Exception as e:
            self.node.get_logger().error(f'Failed to send goal: {e}')
            self.busy = False
            return

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback

        # update movement feedback when available
        self.distance_travelled = getattr(feedback, 'distance_travelled', self.distance_travelled)
        self.angle_turned = getattr(feedback, 'angle_turned', self.angle_turned)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.node.get_logger().error(f'Goal response future exception: {e}')
            self.busy = False
            return

        if not getattr(goal_handle, 'accepted', False):
            # if goal is rejected, log error and set busy to false
            self.node.get_logger().error('Movement Goal rejected')
            self.busy = False
            return

        self.node.get_logger().info('Movement Goal accepted')

        try:
            self.get_result_future = goal_handle.get_result_async()
            self.get_result_future.add_done_callback(self.result_callback)
        except Exception as e:
            self.node.get_logger().error(f'Failed to request result: {e}')
            self.busy = False
            return

    def result_callback(self, future):
        try:
            res = future.result()
            result = getattr(res, 'result', res)
        except Exception as e:
            self.node.get_logger().error(f'Get result future exception: {e}')
            self.busy = False
            return

        success = getattr(result, 'success', None)
        if success is True:
            self.node.get_logger().info('Movement Goal success')
        elif success is False:
            self.node.get_logger().error('Movement Goal fail')
        else:
            # unknown result type; log for debugging
            self.node.get_logger().info(f'Movement Goal result: {result}')

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
            self.inference_callback,
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
