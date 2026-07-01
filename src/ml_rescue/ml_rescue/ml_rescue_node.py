from enum import Enum
import math
# import time

import rclpy
from rclpy.lifecycle import (
    LifecycleNode,
    LifecyclePublisher,
    State,
    TransitionCallbackReturn,
)
from rclpy.subscription import Subscription
from rclpy.timer import Timer
from rescue_msgs.srv import EnableInference, SetRescueState
from robot_msgs.srv import Inference as SendInference
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


class TRescue(LifecycleNode):
    """
    Switches between states within rescue, allowing for better control of resources.

    Lifecycle node
    """

    def __init__(self) -> None:
        super().__init__('ml_rescue')
        self.current_state = States.ENTER
        self.state_started = False

        self.isRobot = False
        self.isActive = False

        self.pub: LifecyclePublisher | None = None
        self.timer: Timer | None = None
        self.vision_sub: Subscription | None = None

        self.rescue_state_srv = self.create_service(
            SetRescueState, 'set_rescue_state', self.set_rescue_state_callback
        )
        self.inference_srv = None
        self.enable_inference = self.create_client(EnableInference, 'enable_inference')
        # self.robot = Movement(self)

        self.dw = 1536
        self.dh = 864
        # self.f_length = 2.75
        self.ball_radius = 0.05  # m
        self.fx = 683.31285  # (self.dw * self.f_length) / 6.54
        self.fy = 683.10689  # (self.dh * self.f_length) / 3.63
        self.cx = 764.89803
        self.cy = 408.4118

        self.obstacle: list = []
        self.target_distance = 0
        self.target_angle = 0

        self.data = None
        self.balls_found = 0

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring ml_rescue node...')
        self.pub = self.create_lifecycle_publisher(String, 'rescue_data', 10)
        self.vision_sub = self.create_subscription(
            Detection2DArray,
            'inference_stream',
            self.inference_callback,
            10,
        )
        self.inference_srv = self.create_service(
            SendInference, 'detections', self.send_inference_data
        )
        self.timer = self.create_timer(0.05, self.state_loop)
        self.timer.cancel()

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating ml_rescue node...')
        self.isActive = True

        self.current_state = States.ENTER
        self.state_started = False
        self.data = None
        self.last_data = None

        if self.timer:
            self.timer.reset()

        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating ml_rescue node...')
        self.set_inference(False)
        self.isActive = False

        if self.timer:
            self.timer.cancel()

        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up ml_rescue node...')
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
        self.get_logger().info('Shutting down ml_rescue node')
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
                if self.isRobot:
                    # self.robot.drive(0.2)
                    pass

                self.transition_to_state(States.SCAN)

        elif self.current_state == States.SCAN:
            # Prescan for all objects OR one ball at a time
            if not self.state_started:
                self.get_logger().info('Enabling inference and scanning for ball')
                self.state_started = True

                # scanning = True
                self.data = []

                self.set_inference(True)

                # while scanning:
                #     if self.data is None or self.data == []:
                #         # Spin robot a little bit
                #         time.sleep(0.1)

                #     else:
                #         # stop spinning
                #         first_object = self.data[0]
                #         self.get_logger().info(f'data is {self.data}')
                #         self.get_logger().info(
                #             f'First object detected is of type: {first_object[0]}, '
                #             f'and there were {len(self.data)} objects detected.'
                #         )

                # self.set_inference(False)

                # self.transition_to_state(States.TARGET_BALL)

        elif self.current_state == States.TARGET_BALL:
            # Move towards ball
            if not self.state_started:
                self.get_logger().info('Targetting a ball')
                self.state_started = True

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
                self.state_started = True

                self.set_inference(True)

                self.transition_to_state(States.DUMP_DROPZONE)

        elif self.current_state == States.DUMP_DROPZONE:
            # Release balls
            if not self.state_started:
                self.get_logger().info('Releasing balls')
                self.state_started = True

                self.set_inference(False)

                self.transition_to_state(States.EXIT)

        elif self.current_state == States.EXIT:
            # Locate exit and turn rescue code off
            if not self.state_started:
                self.get_logger().info('Exiting rescue.....')
                self.state_started = True

                self.set_inference(True)
                if self.isRobot:
                    # self.robot.drive(0.5)
                    pass
                self.set_inference(False)
        else:
            self.get_logger().warn('Invalid rescue state detected')

    def transition_to_state(self, new_state: States):
        self.current_state = new_state
        self.state_started = False

    def inference_callback(self, msg):
        old_data = self.data if self.data is not None else []

        if len(msg.detections) == 0:
            self.get_logger().warn('Nothing detected')
            self.last_data = old_data
            self.data = None
            return

        current_data = []

        for detection in msg.detections:
            # Skip detections with no classification
            if len(detection.results) == 0:
                continue

            result = detection.results[0]
            class_id = result.hypothesis.class_id
            confidence = result.hypothesis.score

            centre_x = detection.bbox.center.position.x
            # centre_y = detection.bbox.center.position.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y

            if class_id in ['ball', 'silver', 'black']:
                average_dimension = (width + height) / 2
                distance = (self.fx * self.ball_radius) / average_dimension

            elif class_id in ['red', 'green']:
                distance = (self.fy * 0.06) / height

            else:
                self.get_logger().warn(f'Class id is not valid: {class_id}')
                continue

            angle = math.atan((centre_x - self.cx) / self.fx)

            # self.get_logger().info(
            #     f'Ball: x={centre_x:.1f}, y={centre_y:.1f}, '
            #     f'w={width:.1f}, h={height:.1f}, conf={confidence:.2f}'
            # )

            self.get_logger().info(
                f'Distance to {class_id}: {distance:.2f}m at angle: {angle:.2f}'
            )

            current_data.append([class_id, confidence, distance, angle, centre_x])

        if current_data == old_data:
            # self.get_logger().warn("Data hasn't changed!!")
            return

        self.last_data = old_data
        self.data = current_data

    def set_inference(self, enabled: bool):

        request = EnableInference.Request()
        request.enabled = enabled

        future = self.enable_inference.call_async(request)

        self.data = None

        return future

    def send_inference_data(self, request, response):
        data = self.data
        self.data = None  # Clear self.data once it is sent, otherwise it may send stale data.
        all_publish_data = []

        if request.message != 'whereball':
            response.success = False
            self.get_logger().warn('The request is not valid')
            return response

        if data is None or len(data) == 0:
            response.success = False
            self.get_logger().warn('No data to send!')
            return response

        for i in data:
            # self.get_logger().info(f'data: {i}')
            if 'silver' in i[0] or 'ball' in i[0]:  # Append silver balls first
                all_publish_data.append(i)
                # self.get_logger().info('Ball detected, sending you a ball')

        for i in data:
            if 'black' in i[0]:  # Ensures black balls are sent after silver balls
                all_publish_data.append(i)

        for i in data:
            if 'green' in i[0] or 'red' in i[0]:  # Ensures evac points are sent after all balls
                all_publish_data.append(i)

        if not all_publish_data:
            self.get_logger().warn('No data to publish!')
            response.success = False
            return response

        for i in range(len(all_publish_data)):
            valid_publish_data = all_publish_data[i]

            response.type.append(valid_publish_data[0])
            response.confidence.append(valid_publish_data[1])
            response.distance.append(valid_publish_data[2])
            response.bearing.append(valid_publish_data[3])
            response.cx.append(valid_publish_data[4])
            response.success = True

        self.get_logger().info(f'Publishing the following: {all_publish_data}')

        return response

    def set_rescue_state_callback(self, request, response):
        try:
            self.transition_to_state(States(request.state))
            response.success = True
            response.message = f'Rescue now in {States(request.state).name} state.'
        except ValueError:
            response.success = False
            response.message = f'Invalid state: {request.state}'
        return response


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


if __name__ == '__main__':
    main()
