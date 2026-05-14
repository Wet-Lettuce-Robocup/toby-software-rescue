from enum import Enum

from lifecycle_msgs.srv import ChangeState  # pyright: ignore[reportMissingImports]
import rclpy  # pyright: ignore[reportMissingImports]
from rclpy.lifecycle import (
    LifecycleNode,
    LifecycleState,
    TransitionCallbackReturn,
)


class State(Enum):
    ENTER = 0
    SCAN = 1
    TARGET_BALL = 2
    TARGET_DROPZONE = 3
    EXIT = 4


class TRescue(LifecycleNode):
    """
    Switches between states within rescue, allowing for better control of resources.

    Lifecycle node
    """

    def __init__(self) -> None:
        super().__init__('ml_rescue')
        self.current_state = State.INIT

        self.balls_found = 0

        # self.rescue_client = self.create_client(ChangeState, 'ml_rescue/change_state')
        # self.motor_client = self.create_client(ChangeState, 'motor_control/change_state')
        # self.camera_client = self.create_client(ChangeState, 'camera_node/change_state')

        self.timer = self.create_timer(0.05, self.state_loop)

    def change_node_state(self, client, transition_id):
        req = ChangeState.Request()
        req.transition.id = transition_id  # example: Transition.TRANSITION_ACTIVATE
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def on_configure(self, state: LifecycleState):
        pass  # hoping that idle button is handled by state machine in robot_core

        return TransitionCallbackReturn.SUCCESS

    def state_loop(self):

        if self.current_state == State.ENTER:
            # Enter the rescue zone with a node
            self.current_state = State.SCAN

        elif self.current_state == State.SCAN:
            # Switch to node to prescan for all objects
            self.current_state = State.TARGET_BALL

        elif self.current_state == State.TARGET_BALL:
            # Switch to ball tracking node
            self.current_state = State.TARGET_DROPZONE

        elif self.current_state == State.TARGET_DROPZONE:
            # Switch to dropzone tracking node
            self.current_state = State.EXIT

        elif self.current_state == State.EXIT:
            # Deactivates all nodes and switches to line following
            # self.change_node_state(self.motor_client, Transition.TRANSITION_DEACTIVATE)
            # self.change_node_state(self.camera_client, Transition.TRANSITION_DEACTIVATE)
            # self.change_node_state(self.rescue_client, Transition.TRANSITION_DEACTIVATE)
            pass


def main(args=None):
    rclpy.init(args=args)
    node = TRescue()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
