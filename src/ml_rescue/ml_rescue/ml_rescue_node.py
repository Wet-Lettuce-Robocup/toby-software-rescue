from enum import Enum

import rclpy
from rclpy.action import ActionClient
from rclpy.lifecycle import (
    LifecycleNode,
    LifecyclePublisher,
    State,
    TransitionCallbackReturn,
)
from rclpy.subscription import Subscription
from rclpy.timer import Timer
from rescue_msgs.srv import EnableInference, SetRescueState
from robot_msgs.action import Move
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray


class States(Enum):
    ENTER = 0
    SCAN = 1
    TARGET_BALL = 2
    GRAB_BALL = 3
    TARGET_DROPZONE = 4
    DUMP_DROPZONE = 5
    EXIT = 6


class Movement:
    """High level movement class that handles robot driving."""

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


class TRescue(LifecycleNode):
    """
    Switches between states within rescue, allowing for better control of resources.

    Lifecycle node
    """

    def __init__(self) -> None:
        super().__init__('ml_rescue')
        self.current_state = States.ENTER
        self.state_started = False

        self.balls_found = 0
        self.isActive = False

        self.pub: LifecyclePublisher | None = None
        self.timer: Timer | None = None
        self.vision_sub: Subscription | None = None

        self.rescue_state_srv = self.create_service(
            SetRescueState, '/set_rescue_state', self.set_rescue_state_callback
        )
        self.enable_inference = self.create_client(EnableInference, '/ml_rescue/enable_inference')
        self.robot = Movement(self)

        self.dw = 1536
        self.dh = 864

    def inference_callback(self, msg):
        self.get_logger().info(f'Recieved: {msg}')
        self.get_logger().info(f'How does {msg} look and is it within {self.dw} {self.dh}')

    def set_inference(self, enabled: bool):

        request = EnableInference.Request()
        request.enabled = enabled

        future = self.enable_inference.call_async(request)

        return future

    def set_rescue_state_callback(self, request, response):
        try:
            self.transition_to_state(States(request.state))
            response.success = True
            response.message = f'Rescue now in {States(request.state).name} state.'
        except ValueError:
            response.success = False
            response.message = f'Invalid state: {request.state}'
        return response

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring rescue node')
        self.pub = self.create_lifecycle_publisher(String, 'rescue_data', 10)
        self.vision_sub = self.create_subscription(
            Detection2DArray,
            '/ml_rescue/inference_stream',
            self.inference_callback,
            10,
        )
        self.timer = self.create_timer(0.05, self.state_loop)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating rescue node')
        self.isActive = True
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating rescue node')
        self.isActive = False
        return super().on_activate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up rescue node')
        if self.timer is not None:
            self.destroy_timer(self.timer)
        if self.pub is not None:
            self.destroy_publisher(self.pub)
        if self.vision_sub is not None:
            self.destroy_subscriber(self.vision_sub)

        self.timer = None
        self.pub = None
        self.vision_sub = None

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down rescue node')
        return TransitionCallbackReturn.SUCCESS

    def state_loop(self):
        if not self.isActive:
            return

        if self.current_state == States.ENTER:
            # Enter the rescue zone
            if not self.state_started:
                self.get_logger().info('Entering rescue zone')
                self.state_started = True

                # move into centre of rescue zone
                self.robot.drive(0.2)

                self.transition_to_state(States.SCAN)

        elif self.current_state == States.SCAN:
            # Prescan for all objects OR one ball at a time
            if not self.state_started:
                self.get_logger().info('Enabling inference and scanning for ball')
                self.state_started = True

                self.set_inference(True)
                self.transition_to_state(States.TARGET_BALL)

        elif self.current_state == States.TARGET_BALL:
            # Move towards ball
            if not self.state_started:
                self.get_logger().info('Targetting a ball')

                self.transition_to_state(States.GRAB_BALL)

        elif self.current_state == States.GRAB_BALL:
            # Pick up ball
            if not self.state_started:
                self.get_logger().info('Grabbing ball')
                self.state_started = True

                self.set_inference(False)

                self.transition_to_state(States.TARGET_DROPZONE)

        elif self.current_state == States.TARGET_DROPZONE:
            # Move towards dropzone
            if not self.state_started:
                self.get_logger().info('Targetting evacuation point')

                self.set_inference(True)

                self.transition_to_state(States.DUMP_DROPZONE)

        elif self.current_state == States.DUMP_DROPZONE:
            # Release balls
            if not self.state_started:
                self.get_logger().info('Releasing balls')

                self.set_inference(False)

                self.transition_to_state(States.EXIT)

        elif self.current_state == States.EXIT:
            # Locate exit and turn rescue code off
            if not self.state_started:
                self.get_logger().info('Exiting rescue.....')

                self.set_inference(True)
                self.robot.drive(0.5)
                self.set_inference(False)
        else:
            self.get_logger().warn('Invalid rescue state detected')

    def transition_to_state(self, new_state: States):
        self.current_state = new_state
        self.state_started = False


def main(args=None):
    rclpy.init(args=args)
    node = TRescue()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down rescue node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.out.release()


if __name__ == '__main__':
    main()
