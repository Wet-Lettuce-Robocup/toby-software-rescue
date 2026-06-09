from enum import Enum

from geometry_msgs.msg import Twist
from lifecycle_msgs.srv import ChangeState
import rclpy
from rclpy.lifecycle import (
    LifecycleNode,
    LifecyclePublisher,
    State,
    TransitionCallbackReturn,
)
from rclpy.publisher import Publisher
from rclpy.timer import Timer
from rescue_msgs.srv import SetRescueState
from std_msgs.msg import String


class States(Enum):
    ENTER = 0
    SCAN = 1
    TARGET_BALL = 2
    GRAB_BALL = 3
    TARGET_DROPZONE = 4
    DUMP_DROPZONE = 5
    EXIT = 6


class TRescue(LifecycleNode):
    """
    Switches between states within rescue, allowing for better control of resources.

    Lifecycle node
    """

    def __init__(self) -> None:
        super().__init__('ml_rescue')
        self.current_state = States.ENTER
        self.state_is_active = False

        self.balls_found = 0
        self.isActive = False

        self.twist_pub: Publisher | None = None
        self.pub: LifecyclePublisher | None = None
        self.timer: Timer | None = None

        self.rescue_state_srv = self.create_service(
            SetRescueState, 'set_rescue_state', self.set_rescue_state_callback
        )

    def set_rescue_state_callback(self, request, response):
        try:
            self.transition_to_state(States(request.state))
            response.success = True
            response.message = f'Rescue now in {States(request.state).name} state.'
        except ValueError:
            response.success = False
            response.message = f'Invalid state: {request.state}'
        return response

    def change_node_state(self, client, transition_id):
        req = ChangeState.Request()
        req.transition.id = transition_id  # example: Transition.TRANSITION_ACTIVATE
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring rescue node')
        self.pub = self.create_lifecycle_publisher(String, 'rescue_data', 10)
        self.twist_pub = self.create_publisher(Twist, 'cmd_vel', 10)
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
        if self.twist_pub is not None:
            self.destroy_publisher(self.twist_pub)

        self.timer = None
        self.pub = None
        self.twist_pub = None

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down rescue node')
        return TransitionCallbackReturn.SUCCESS

    def state_loop(self):
        if not self.isActive:
            return

        if self.current_state == States.ENTER:
            # Enter the rescue zone
            if not self.state_is_active:
                self.get_logger().info('Entering rescue zone')
                self.state_is_active = True

                # move into centre of rescue zone
                twist = Twist()
                twist.linear.x = 0.2
                self.twist_pub.publish(twist)  # Theoretically makes robot move forwards
                self.current_state = States.SCAN

        elif self.current_state == States.SCAN:
            # Prescan for all objects OR one ball at a time
            self.current_state = States.TARGET_BALL

        elif self.current_state == States.TARGET_BALL:
            # Move towards ball
            self.current_state = States.GRAB_BALL

        elif self.current_state == States.GRAB_BALL:
            # Pick up ball
            self.current_state = States.TARGET_DROPZONE

        elif self.current_state == States.TARGET_DROPZONE:
            # Move towards dropzone
            self.current_state = States.DUMP_DROPZONE

        elif self.current_state == States.DUMP_DROPZONE:
            # Release balls
            self.current_state = States.EXIT

        elif self.current_state == States.EXIT:
            # Locate exit and turn rescue code off
            pass

    def transition_to_state(self, new_state: States):
        self.current_state = new_state
        self.state_is_active = False


def main(args=None):
    rclpy.init(args=args)
    node = TRescue()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
