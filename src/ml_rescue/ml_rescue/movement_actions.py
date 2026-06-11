from robot_msgs.action import Move
from rclpy.action import ActionClient


class Movement:
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

        self.get_result_future = goal_handle.get_result_async()  #
        self.get_result_future.add_done_callback(self.get_result_callback)

    def result_callback(self, future):
        result = future.result().result

        if result.success:
            self.get_logger().info('Movement Goal success')
        else:
            self.get_logger().error('Movement Goal fail')

        self.busy = False
