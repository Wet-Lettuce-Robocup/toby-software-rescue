from enum import Enum

from geometry_msgs.msg import Twist
from lifecycle_msgs.srv import ChangeState
import rclpy
from rclpy.lifecycle import (
    LifecycleNode,
    LifecycleState,
    TransitionCallbackReturn,
)
from std_msgs.msg import String


class State(Enum):
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
        self.current_state = State.ENTER
        self.state_is_active = False

        self.balls_found = 0

        self.twist_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.pub = None
        self.timer = None

    def change_node_state(self, client, transition_id):
        req = ChangeState.Request()
        req.transition.id = transition_id  # example: Transition.TRANSITION_ACTIVATE
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    # def on_configure(self, state: LifecycleState):
    #     self.pub = self.create_lifecycle_publisher(String, 'rescue_data', 10)
    #     self.timer = self.create_timer(0.05, self.state_loop)
    #     return TransitionCallbackReturn.SUCCESS

    # def on_activate(self, state: LifecycleState):
    #     self.get_logger().info('Activating rescue code')
    #     return TransitionCallbackReturn.SUCCESS

    # def on_deactivate(self, state: LifecycleState):
    #     self.get_logger().info('Deactivating rescue code')
    #     return TransitionCallbackReturn.SUCCESS

    # def on_cleanup(self, state: LifecycleState):
    #     self.get_logger().info('Cleaning up rescue code')
    #     self.destroy_timer(self.timer)
    #     self.destroy_publisher(self.pub)
    #     return TransitionCallbackReturn.SUCCESS

    def state_loop(self):

        if self.current_state == State.ENTER:
            # Enter the rescue zone
            if not self.state_is_active:
                self.get_logger().info('Entering rescue zone')
                self.state_is_active = True

                # move into centre of rescue zone
                twist = Twist()
                twist.linear.x = 0.2
                self.twist_pub.publish(twist)  # Theoretically makes robot move forwards
                self.current_state = State.SCAN

        elif self.current_state == State.SCAN:
            # Prescan for all objects OR one ball at a time
            self.current_state = State.TARGET_BALL

        elif self.current_state == State.TARGET_BALL:
            # Move towards ball
            self.current_state = State.GRAB_BALL

        elif self.current_state == State.GRAB_BALL:
            # Pick up ball
            self.current_state = State.TARGET_DROPZONE

        elif self.current_state == State.TARGET_DROPZONE:
            # Move towards dropzone
            self.current_state = State.DUMP_DROPZONE

        elif self.current_state == State.DUMP_DROPZONE:
            # Release balls
            self.current_state = State.EXIT

        elif self.current_state == State.EXIT:
            # Locate exit and turn rescue code off
            pass

    def transition_to_state(self, new_state: State):
        self.current_state = new_state
        self.state_is_active = False


def main(args=None):
    rclpy.init(args=args)
    node = TRescue()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
