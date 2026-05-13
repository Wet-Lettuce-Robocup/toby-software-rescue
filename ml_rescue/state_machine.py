from enum import Enum

import rclpy# pyright: ignore[reportMissingImports]
from rclpy.node import Node# pyright: ignore[reportMissingImports]
from std_msgs.msg import String, Bool# pyright: ignore[reportMissingImports]
from geometry_msgs.msg import Twist # pyright: ignore[reportMissingImports]
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn # pyright: ignore[reportMissingImports]
from lifecycle_msgs.srv import ChangeState # pyright: ignore[reportMissingImports]
from lifecycle_msgs.msg import Transition # pyright: ignore[reportMissingImports]



class State(Enum):
    INIT = 0
    ENTER = 1
    SCAN = 2
    TARGET_BALL = 3
    TARGET_DROPZONE = 4
    EXIT = 5

class TRescue(LifecycleNode):
    """
    Switches between states within rescue, allowing for better control of resources.    
    """
    
    def __init__(self) -> None:
        super().__init__('ml_rescue')
        self.current_state = State.INIT
        
        self.balls_found = 0
        
        self.rescue_client = self.create_client(ChangeState, 'ml_rescue/change_state')
        self.motor_client = self.create_client(ChangeState, 'motor_control/change_state')
        self.camera_client = self.create_client(ChangeState, 'camera_node/change_state')
        
        self.timer = self.create_timer(0.05, self.state_loop)
       
        
    def change_node_state(self, client, transition_id):
        req = ChangeState.Request()
        req.transition.id = transition_id #example: Transition.TRANSITION_ACTIVATE
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        
    def on_configure(self, state:LifecycleState):
        pass #hoping that idle button is handled by state machine in robot_core
        
        return TransitionCallbackReturn.SUCCESS
    
    
    def state_loop(self):
        if self.current_state == State.INIT:
            # Initialisation of motors 
            self.change_node_state(self.motor_client, Transition.TRANSITION_ACTIVATE)
            self.current_state = State.ENTER
            
        elif self.current_state == State.ENTER:
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
            self.change_node_state(self.motor_client, Transition.TRANSITION_DEACTIVATE)
            self.change_node_state(self.camera_client, Transition.TRANSITION_DEACTIVATE)
            self.change_node_state(self.rescue_client, Transition.TRANSITION_DEACTIVATE)
            
            

def main(args=None):
    rclpy.init(args=args)
    node = TRescue()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    