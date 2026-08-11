from enum import Enum
import math

from geometry_msgs.msg import Twist
import rclpy
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.lifecycle import (
    LifecycleNode,
    State,
    TransitionCallbackReturn,
)
from rclpy.publisher import Publisher
from rclpy.service import Service
from rclpy.subscription import Subscription
from rclpy.timer import Timer
from rescue_msgs.srv import EnableInference, InferenceDetections, SetRescueState
from robot_msgs.action import MoveTime
from robot_msgs.srv import Inference as SendInference
from robot_msgs.srv import ServoCommand
from std_msgs.msg import Bool, Int32


class States(Enum):
    """A class that holds the states for rescue."""

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

    # Setup for camera calibration and distance estimation
    dw = 1536
    dh = 864
    # self.f_length = 2.75
    ball_radius = 0.05  # m
    fx = 683.31285  # (self.dw * self.f_length) / 6.54
    fy = 683.10689  # (self.dh * self.f_length) / 3.63
    cx = 764.89803
    cy = 408.4118

    def __init__(self) -> None:
        super().__init__('ml_rescue')
        self.current_state = States.ENTER
        self.state_started = False
        self.sub_state = 0

        self.isActive = False

        # Initialise ros topics and services

        self.robot: Movement | None = None

        self.timer: Timer | None = None

        self.cmd_vel_pub: Publisher | None = None
        self.rescue_active_pub: Publisher | None = None
        self.led_pub: Publisher | None = None

        self.front_tof_subscriber: Subscription | None = None
        self.claw_tof_subscriber: Subscription | None = None
        self.side_tof_subscriber: Subscription | None = None

        self.rescue_state_srv: Service | None = None
        self.inference_srv: Service | None = None
        self.rescue_detections_cli: Client | None = None
        self.enable_inference_cli: Client | None = None

        self.claw_pub: Client | None = None
        self.lift_pub: Client | None = None
        self.gate_pub: Client | None = None

        # Setup variables for data processing
        self.data = None
        self.last_data = None
        self.inference_returned = False
        self.target_object = None
        # self.obstacles: list[list] = []
        self.balls_found = 0
        self.target_dropzone = None

        # Set tof data to None in case sensors fail to initialise
        self.front_tof_dist = None
        self.claw_tof_dist = None
        self.side_tof_dist = None

        self.servo_busy = False
        self.inference_request_pending = False

        self.servo_available_time = 0
        self.servo_timeout = 0
        self.target_timestamp = 0

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring ml_rescue node...')

        self.robot = Movement(self)

        self.timer = self.create_timer(0.05, self.state_loop)
        self.timer.cancel()

        self.claw_pub = self.create_client(ServoCommand, '/servo/grab')
        self.lift_pub = self.create_client(ServoCommand, '/servo/lift')
        self.gate_pub = self.create_client(ServoCommand, '/servo/tray_release')

        self.rescue_state_srv = self.create_service(
            SetRescueState, 'set_rescue_state', self.set_rescue_state_callback
        )
        self.inference_srv = self.create_service(
            SendInference, 'detections', self.send_inference_data
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rescue_active_pub = self.create_publisher(Bool, '/rescue_active', 10)
        self.led_pub = self.create_publisher(Int32, '/front_led/target_brightness', 10)

        self.front_tof_subscriber = self.create_subscription(
            Int32, '/tof/front', self.front_tof_callback, 10
        )
        self.claw_tof_subscriber = self.create_subscription(
            Int32, '/tof/claw', self.claw_tof_callback, 10
        )
        self.side_tof_subscriber = self.create_subscription(
            Int32, '/tof/side', self.side_tof_callback, 10
        )

        self.enable_inference_cli = self.create_client(EnableInference, 'enable_inference')
        self.rescue_detections_cli = self.create_client(InferenceDetections, 'rescue_detections')

        while not self.rescue_detections_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for rescue_detections service...')

        self.get_logger().info('Configuring complete')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating ml_rescue node...')
        self.isActive = True

        self.current_state = States.ENTER
        self.state_started = False
        self.sub_state = 0
        self.data = None
        self.last_data = None
        self.balls_found = 0
        self.target_dropzone = None
        self.front_tof_dist = None
        self.claw_tof_dist = None
        self.side_tof_dist = None

        self.servo_busy = False
        self.inference_request_pending = False

        self.servo_available_time = 0
        self.servo_timeout = 0
        self.target_timestamp = 0

        if self.timer:
            self.timer.reset()

        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating ml_rescue node...')
        self.isActive = False
        self.set_inference(False)

        if self.timer:
            self.timer.cancel()

        self.stop_moving()
        if self.led_pub is not None:
            self.led_pub.publish(Int32(data=0))

        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up ml_rescue node...')
        if self.timer is not None:
            self.destroy_timer(self.timer)

        self.timer = None

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down ml_rescue node')
        return TransitionCallbackReturn.SUCCESS

    def state_loop(self):
        """Core loop which runs through states and executes main logic tasks."""
        if not self.isActive:
            return
        else:
            now = self.get_clock().now()

        if self.current_state == States.ENTER:
            # Enter the rescue zone
            if not self.state_started:
                self.get_logger().info('Entering rescue zone')
                self.state_started = True

                self.stop_moving()

                # Wait 2 seconds without blocking state_loop
                self.target_timestamp = now + rclpy.duration.Duration(seconds=2.0)

                self.sub_state = 1

            elif now >= self.target_timestamp and self.sub_state == 1:
                # move into centre of rescue zone
                self.set_inference(True)

                self.lift('up')
                self.sub_state = 2

            elif self.sub_state == 2 and not self.servo_busy and now >= self.servo_available_time:
                self.claw('close')

                self.sub_state = 3

            elif (
                self.sub_state == 3
                and not self.robot.busy
                and not self.servo_busy
                and now >= self.servo_available_time
            ):
                # Send drive command and wait for it to return without blocking state_loop
                self.robot.drive(0.2)
                self.sub_state = 4

            # elif self.sub_state == 3:
            #     if self.robot.busy:
            #         return

            #     self.robot.drive(0, 45)
            #     self.sub_state = 4

            elif self.sub_state == 4 and not self.robot.busy:
                self.get_logger().info('Entered rescue zone.')
                self.transition_to_state(States.SCAN)

        elif self.current_state == States.SCAN:
            # Scan for either balls or evacuation points
            if not self.state_started:
                self.get_logger().info('Starting scanning...')
                self.state_started = True

                self.led_pub.publish(Int32(data=30))

                if self.balls_found == 3:
                    # If all balls are found, search for evacuation points
                    self.request_inference('evacpoint')
                    self.get_logger().info('Requesting point')
                else:
                    self.request_inference('ball')

                if self.data is None or self.data == []:
                    # Spin robot a little bit to seek out more balls
                    self.start_moving(0, 0.02)
                    return

            if not self.inference_returned:
                # Wait for inference data without blocking state_loop
                return

            current_data = self.data
            self.data = None

            if current_data is None or len(current_data) == 0:
                # Request inference data again if nothing found

                # self.get_logger().info('No objects detected, requesting inference again...')
                if self.balls_found == 3 and not self.inference_request_pending:
                    self.request_inference('evacpoint')
                    self.get_logger().info('Requesting point')

                else:
                    self.request_inference('ball')
                return

            self.get_logger().info(f'{len(current_data)} objects detected.')

            self.target_object = None

            # Iterate through returned data and only target first detection
            # In future, save all objects and navigate to each using odometry, once it's fixed
            for i in current_data:
                obj_type = i[0]
                if self.balls_found == 3:
                    # If all balls found, target green dropzone, then red if no green is found
                    if obj_type == 'green' and self.target_dropzone is None:
                        self.target_dropzone = 'green'
                        self.target_object = i
                    elif obj_type == 'red' and self.target_dropzone == 'green':
                        self.target_dropzone = 'red'
                        self.target_object = i
                elif (obj_type == 'silver' and self.balls_found in [0, 1]) or (
                    obj_type == 'black' and self.balls_found == 2
                ):
                    # If not all balls found, target silver, then black if both silvers are found
                    self.target_object = i
                else:
                    self.get_logger().warn(
                        f'Something is broken: {obj_type} with {self.balls_found} balls rescued'
                    )
                    # In future add handling for multiple obstacles for avoidance

            if self.target_object is not None:
                # stop spinning if target acquired
                self.stop_moving()
                if self.balls_found < 3:
                    self.transition_to_state(States.TARGET_BALL)
                else:
                    self.transition_to_state(States.TARGET_DROPZONE)

        elif self.current_state == States.TARGET_BALL:
            # Move towards ball using distance and angle estimation

            bearing = self.target_object[3] - 5
            distance = self.target_object[2] - 0.05

            if not self.state_started:
                self.get_logger().info('Targeting a ball...')
                self.state_started = True

                self.claw('open')
                self.sub_state = 1

            elif (
                not self.robot.busy
                and self.sub_state == 1
                and not self.servo_busy
                and now >= self.servo_available_time
            ):
                self.robot.drive(0, bearing)  # need to test and see how far robot turns
                self.sub_state = 2

            elif self.sub_state == 2 and not self.robot.busy:
                self.get_logger().info('Robot is facing ball')

                self.robot.drive(distance)
                self.sub_state = 3

            elif self.sub_state == 3 and not self.robot.busy:
                # TODO: Add a check to make sure ball is in correct spot before picking up

                self.lift('down')
                self.sub_state = 4

            elif self.sub_state == 4 and not self.servo_busy and now >= self.servo_available_time:
                self.get_logger().info('Robot is at ball')

                self.transition_to_state(States.GRAB_BALL)

        elif self.current_state == States.GRAB_BALL:
            # Pick up ball
            if not self.state_started and not self.robot.busy:
                self.get_logger().info('Grabbing ball...')
                self.state_started = True

                self.robot.drive(0.1, velocity=50)

                self.sub_state = 0

            if self.sub_state == 0 and not self.robot.busy and not self.servo_busy:
                self.claw('close')
                self.balls_found += 1

                self.sub_state = 1

            elif (
                self.sub_state == 1
                and not self.robot.busy
                and not self.servo_busy
                and now >= self.servo_available_time
            ):
                self.get_logger().info('Reversing...')
                self.robot.drive(-0.1)
                self.sub_state = 2

            elif self.sub_state == 2 and not self.robot.busy:
                self.lift('up')
                self.sub_state = 3

            elif self.sub_state == 3 and not self.servo_busy and now >= self.servo_available_time:
                # TODO: Add a check to make sure ball is actually picked up (limit switch)

                self.target_timestamp = now + rclpy.duration.Duration(seconds=3.0)
                self.sub_state = 4

            elif now >= self.target_timestamp and self.sub_state == 4:
                self.get_logger().info('Back to scanning')
                self.transition_to_state(States.SCAN)

        elif self.current_state == States.TARGET_DROPZONE:
            # Move towards dropzone using distance/angle estimation

            bearing = self.target_object[3]
            distance = self.target_object[2] - 0.05

            if not self.state_started:
                self.get_logger().info('Targeting evacuation point')
                self.state_started = True

                self.sub_state = 0

            elif self.sub_state == 0 and not self.robot.busy:
                self.robot.drive(0, bearing)
                self.sub_state = 1

            elif self.sub_state == 1 and not self.robot.busy:
                self.robot.drive(distance)
                self.sub_state = 2

            elif self.sub_state == 2 and not self.robot.busy:
                self.transition_to_state(States.DUMP_DROPZONE)

        elif self.current_state == States.DUMP_DROPZONE:
            # Release balls into dropzone

            if not self.state_started and not self.robot.busy:
                self.get_logger().info('Releasing balls')
                self.state_started = True

                self.robot.drive(-0.1)
                self.sub_state = 1

            elif self.sub_state == 1 and not self.robot.busy:
                self.robot.drive(0, 180)
                self.sub_state = 2

            elif self.sub_state == 2 and not self.robot.busy:
                self.robot.drive(-0.15)
                self.sub_state = 3

            elif self.sub_state == 3 and not self.robot.busy and not self.servo_busy:
                if self.target_dropzone == 'red':
                    self.claw('open')
                self.sub_state = 4

            elif self.sub_state == 4 and not self.servo_busy and now >= self.servo_available_time:
                self.gate('open')
                self.sub_state = 5

            elif self.sub_state == 5 and not self.servo_busy and now >= self.servo_available_time:
                self.target_timestamp = now + rclpy.duration.Duration(seconds=3.0)
                self.sub_state = 6

            elif now >= self.target_timestamp and self.sub_state == 6 and not self.servo_busy:
                self.gate('close')
                self.sub_state = 7

            elif self.sub_state == 7 and not self.servo_busy and now >= self.servo_available_time:
                self.claw('close')
                self.sub_state = 8

            elif self.sub_state == 8 and not self.servo_busy and now >= self.servo_available_time:
                if self.target_dropzone == 'green':
                    self.transition_to_state(States.SCAN)
                elif self.target_dropzone == 'red':
                    self.transition_to_state(States.EXIT)
                else:
                    self.get_logger().info('error after dropping off balls')

        elif self.current_state == States.EXIT:
            # Locate exit using maze algorithm and turn rescue code off
            if not self.state_started:
                self.get_logger().info('Exiting rescue.....')
                self.set_inference(False)
                self.state_started = True

                self.led_pub.publish(Int32(data=0))

                self.sub_state = 0

            elif self.sub_state == 0 and not self.robot.busy:
                self.robot.drive(0.2)
                self.sub_state = 1

            elif self.sub_state == 1 and not self.robot.busy:
                status, angle = self.left_wall_follow()

                if status == 'exit':
                    self.get_logger().info('Deactivating rescue')
                    self.rescue_active_pub.publish(Bool(data=False))
                    # Add check for black line
                elif status == 'right':
                    self.stop_moving()
                    self.robot.drive(0, 90)
                    self.sub_state = 2
                elif status == 'wall':
                    self.start_moving(0.2, angle)
                elif status == 'error':
                    self.get_logger().warn('No tof data for wall following')
                    self.stop_moving()

            elif self.sub_state == 2 and not self.robot.busy:
                self.sub_state = 1

        else:
            self.get_logger().warn('Invalid rescue state detected')

    def left_wall_follow(self, target_distance=0.15):
        # Turn right if wall in front, exit rescue if no wall to left, else go straight
        left_dist = self.side_tof_dist
        front_dist = self.front_tof_dist

        if front_dist is None or left_dist is None:
            return 'error', 0.0

        if front_dist < 0.2:
            return 'right', 0.0
        if left_dist >= 0.4:
            return 'exit', 0.0

        error = target_distance - left_dist

        angle = error * 3
        angle = max(-1.5, min(1.5, angle))

        return 'wall', angle

    def request_inference(self, message):
        if self.inference_request_pending:
            return
        # Request inference data from vision_node
        request = InferenceDetections.Request()

        request.message = message

        self.inference_returned = False
        self.inference_request_pending = True

        future = self.rescue_detections_cli.call_async(request)
        future.add_done_callback(self._parse_results)

    def _parse_results(self, future):
        self.inference_request_pending = False
        # Function to handle results and estimate distance
        try:
            msg = future.result()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            self.data = None
            self.inference_returned = True
            return

        old_data = self.data if self.data is not None else []

        if not msg.success:
            # self.get_logger().info('inference returned false')
            self.data = None
            self.inference_returned = True
            return

        self.inference_returned = True

        # if len(msg.detections.detections) == 0:
        #     # self.get_logger().warn('Nothing detected')
        #     self.last_data = old_data
        #     self.data = None
        #     return

        current_data = []

        for detection in msg.detections.detections:
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

            angle = math.degrees(math.atan((centre_x - self.cx) / self.fx))

            # self.get_logger().info(
            #     f'Ball: x={centre_x:.1f}, y={centre_y:.1f}, '
            #     f'w={width:.1f}, h={height:.1f}, conf={confidence:.2f}'
            # )

            self.get_logger().info(
                f'Distance to {class_id}: {distance:.2f}m at angle: {angle:.2f}'
            )

            current_data.append([class_id, confidence, distance, angle, centre_x])

        if current_data == old_data:
            self.get_logger().warn("Data hasn't changed!!")
            return

        self.last_data = old_data
        self.data = current_data

    def set_inference(self, enabled: bool):

        request = EnableInference.Request()
        request.enabled = enabled

        if self.enable_inference_cli:
            future = self.enable_inference_cli.call_async(request)

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

        for valid_publish_data in all_publish_data:
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

    def start_moving(self, linear_x=0.0, angular_z=0.0):
        if self.cmd_vel_pub is None:
            self.get_logger().warn('Movement command called, but cmd_vel_pub does not exist!')
            return
        self.get_logger().info(f'Movement command called with lx {linear_x} and az {angular_z}')
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def stop_moving(self):
        self.start_moving(0, 0)

    def claw(self, state):
        """State is either 'open' or 'close'."""
        if state == 'open':
            angle = 0.5
        elif state == 'close':
            angle = 1.0
        else:
            self.get_logger().warn('Invalid grab command')
            return

        self.servo_request('claw_pub', angle)

    def lift(self, state):
        """State is either 'up' or 'down'."""
        if state == 'up':
            angle = 2.7
        elif state == 'down':
            angle = 0.4
        else:
            self.get_logger().warn('Invalid lift command')
            return

        self.servo_request('lift_pub', angle)

    def gate(self, state):
        """State is either 'open' or 'close'."""
        if state == 'open':
            angle = 2.3
        elif state == 'close':
            angle = 0.8
        else:
            self.get_logger().warn('Invalid gate command')
            return

        self.servo_request('gate_pub', angle)

    def servo_request(self, servo, angle):
        self.get_logger().info(f'Servo {servo} requested')
        self.servo_busy = True

        request = ServoCommand.Request()
        request.angle = angle

        client = getattr(self, servo)
        future = client.call_async(request)

        future.add_done_callback(self.servo_callback)

    def servo_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(
                f'Servo call result: {response.message} success = {response.success}'
            )

            if response.success:
                self.servo_busy = False
                self.servo_available_time = self.get_clock().now() + rclpy.duration.Duration(
                    seconds=1.0
                )
                self.servo_timeout = self.get_clock().now() + rclpy.duration.Duration(seconds=5.0)
            else:
                self.get_logger().error(f'Servo call failed: {response.message}')
                self.servo_busy = False

        except Exception as e:
            self.get_logger().error(f'Servo service failed: {e}')
            self.servo_busy = False

    def front_tof_callback(self, msg):
        try:
            self.front_tof_dist = float(msg.data) / 1000.0
        except Exception as e:
            self.get_logger().warn(f'Front tof error {e} with msg {msg}')
            self.front_tof_dist = None

    def claw_tof_callback(self, msg):
        try:
            self.claw_tof_dist = float(msg.data) / 1000.0
        except Exception as e:
            self.get_logger().warn(f'Claw tof error {e} with msg {msg}')
            self.claw_tof_dist = None

    def side_tof_callback(self, msg):
        try:
            self.side_tof_dist = float(msg.data) / 1000.0
        except Exception as e:
            self.get_logger().warn(f'Side tof error {e} with msg {msg}')
            self.side_tof_dist = None

    def transition_to_state(self, new_state: States):
        self.current_state = new_state
        self.state_started = False
        self.sub_state = 0

        if new_state == States.SCAN:
            self.data = None
            self.inference_returned = False


class Movement:
    """Class that handles higher level robot movement."""

    def __init__(self, node):
        self.node = node
        self._move_client = ActionClient(node, MoveTime, '/move_time')
        self.busy = False
        self.current_angle = 0.0
        self.distance_travelled = 0.0
        self.angle_turned = 0.0
        self._last_goal_vel = 0.0
        self._last_goal_angular_vel = 0.0
        self._last_goal_time = 0.0
        self._sequence = None
        self._on_complete = None

    def drive(self, distance, angle=0, velocity=100):
        self.node.get_logger().info(f'Drive called with distance {distance} and angle {angle}')
        distance *= 510
        angle *= 2.1
        linear_time = abs(distance) / abs(velocity) if velocity != 0 and distance != 0 else 0.0
        angular_time = abs(angle) / abs(velocity) if velocity != 0 and angle != 0 else 0.0
        time_required = max(linear_time, angular_time)

        if time_required <= 0.0:
            self.node.get_logger().warn('Drive called with zero distance and angle; ignoring')
            return

        goal = MoveTime.Goal()
        goal.time = float(time_required)
        linear_vel = math.copysign(velocity, distance) if distance != 0 else 0.0
        angular_vel = math.copysign(velocity, angle) if angle != 0 else 0.0
        goal.vel = float(linear_vel)
        goal.angular_vel = float(angular_vel)
        self._last_goal_vel = float(linear_vel)
        self._last_goal_angular_vel = float(angular_vel)
        self._last_goal_time = float(time_required)
        self.busy = True

        try:
            available = self._move_client.wait_for_server(timeout_sec=2.0)
        except Exception as e:
            self.node.get_logger().error(f'wait_for_server exception: {e}')
            self.busy = False
            return

        if not available:
            self.node.get_logger().error('Action server "move_time" not available (timeout)')
            self.busy = False
            return

        try:
            self.send_goal_future = self._move_client.send_goal_async(
                goal, feedback_callback=self.feedback_callback
            )
            self.send_goal_future.add_done_callback(self.goal_response_callback)
        except Exception as e:
            self.node.get_logger().error(f'Failed to send goal: {e}')
            self.busy = False
            return

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        time_elapsed = getattr(feedback, 'time_elapsed', None)
        if time_elapsed is not None:
            te = float(time_elapsed)
            te_sec = te * 1e-9 if te > 1e6 else te
            self.distance_travelled = self._last_goal_vel * te_sec
            self.angle_turned = self._last_goal_angular_vel * te_sec
        else:
            self.distance_travelled = getattr(
                feedback, 'distance_travelled', self.distance_travelled
            )
            self.angle_turned = getattr(feedback, 'angle_turned', self.angle_turned)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.node.get_logger().error(f'Goal response future exception: {e}')
            self.busy = False
            return

        if not getattr(goal_handle, 'accepted', False):
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
            # self.node.get_logger().info('Movement Goal success')
            pass
        elif success is False:
            self.node.get_logger().error('Movement Goal fail')
        else:
            self.node.get_logger().info(f'Movement Goal result: {result}')

        self.busy = False
        if self._on_complete is not None:
            self._on_complete()

    def _advance_sequence(self):
        if self._sequence is None:
            return
        try:
            next(self._sequence)
        except StopIteration:
            self._sequence = None
            self._on_complete = None

    def run_sequence(self, sequence_gen):
        self._sequence = sequence_gen
        self._on_complete = self._advance_sequence
        self._advance_sequence()


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
