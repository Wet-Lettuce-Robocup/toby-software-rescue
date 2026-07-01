from functools import partial
import os
import queue
import time
# import subprocess


from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from hailo_platform import VDevice
import numpy as np
import rclpy
from rclpy.node import Node
from rescue_msgs.srv import EnableInference
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class VisionNode(Node):
    """
    A node.

    - has things
    """

    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('raw_image_topic', '/front_camera/camera_node/image_raw')
        self.declare_parameter('ml_rescue_debug', True)

        self.debug = self.get_parameter('ml_rescue_debug').value
        self.get_logger().info(f'Vision node debug status: {self.debug}')

        self.camera_sub = self.create_subscription(
            Image,
            self.get_parameter('raw_image_topic').value,
            self.image_callback,
            1,
        )
        self.inference_pub = self.create_publisher(Detection2DArray, 'inference_stream', 10)
        self.rescue_active_srv = self.create_service(
            EnableInference, 'enable_inference', self.rescue_active_callback
        )

        self.fps = 30
        self.create_timer(1 / self.fps, self.inference_callback)

        self.isActive = False
        self.inferenceBusy = False

        self.bridge = CvBridge()

        self.latest_image = None
        self.latest_image_header = None

        self.hailo = 'robotyolov8s'  # Model name
        self.hef_path = os.path.join(
            get_package_share_directory('ml_rescue'), 'modelhef', f'{self.hailo}.hef'
        )
        self.imgsz = 640
        self.conf_threshold = 0.8
        self.model_classes = ['black', 'silver']

        self.results_queue = queue.Queue(maxsize=2)

        try:
            self.target = VDevice()
            self.infer_model = self.target.create_infer_model(self.hef_path)
            self.input_name = self.infer_model.input_names[0]
            self.output_name = self.infer_model.output_names[0]
            self.output_shape = self.infer_model.output(self.output_name).shape
            self.configured_model = self.infer_model.configure()

        except Exception as e:
            self.get_logger().error(f'Error loading hailort: {e}')
            self.target = None
            raise

        self.out = None

        self.dw = 1536
        self.dh = 864

        pipeline = (
            'appsrc ! queue '
            f'! video/x-raw,format=BGR,width={self.dw},height={self.dh},framerate={self.fps}/1 '
            '! videoconvert '
            '! x264enc bitrate=8000 tune=zerolatency speed-preset=fast '
            '! mp4mux fragment-duration=1000 ! filesink location=/videos/output_video.mp4'
        )
        self.out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, self.fps, (self.dw, self.dh))
        self.get_logger().info(f'Video writer opened: {self.out.isOpened()}')

    def rescue_active_callback(self, request, response):
        self.isActive = request.enabled
        response.message = (
            'Inference enabled successfully' if request.enabled else 'Inference disabled'
        )

        self.get_logger().info(response.message)

        return response

    def image_callback(self, msg):
        self.latest_image_header = msg.header
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def inference_callback(self):
        if not self.isActive:
            return

        if self.latest_image is None:
            self.get_logger().warn('No image received, is the front camera working?')
            return

        self.run_inference()  # What I want to do is separate inference from red/green contours to hopefully boost fps

    def run_inference(self):

        # start_time = time.time()

        image_header = self.latest_image_header
        raw_frame = self.latest_image.copy()

        # self.out.write(raw_frame)
        resized_frame = cv2.resize(raw_frame, (self.imgsz, self.imgsz))
        input_data = np.ascontiguousarray(resized_frame)

        bindings = self.configured_model.create_bindings()
        bindings.input(self.input_name).set_buffer(input_data)

        output_buffer = np.zeros(self.output_shape, dtype=np.float32)
        bindings.output(self.output_name).set_buffer(output_buffer)

        bound_callback = partial(
            self._inference_callback,
            output_buffer=output_buffer,
            display_frame=raw_frame,
        )

        # self.configured_model.wait_for_async_ready(timeout_ms=1000)

        # if self.inferenceBusy:
        #     return

        # self.inferenceBusy = True

        job = self.configured_model.run_async([bindings], bound_callback)
        job.wait(1000)

        try:
            vis_frame, latest_balls = self.results_queue.get_nowait()

            self.get_logger().info(f'balls: {latest_balls}')

            detection_msg = Detection2DArray()
            detection_msg.header.stamp = image_header.stamp
            detection_msg.header.frame_id = image_header.frame_id

            for ball in latest_balls:
                # self.get_logger().info('Ball detected')
                y1, x1, y2, x2 = ball['box']
                score = ball['score']

                # Scale values back to original frame size
                px1 = int(x1 * self.dw)
                py1 = int(y1 * self.dh)
                px2 = int(x2 * self.dw)
                py2 = int(y2 * self.dh)

                # Clamp boxes inside your image frames
                px1, px2 = max(0, min(self.dw, px1)), max(0, min(self.dw, px2))
                py1, py2 = max(0, min(self.dh, py1)), max(0, min(self.dh, py2))

                pxc = (px1 + px2) / 2
                pyc = (py1 + py2) / 2

                # self.get_logger().info(f'pxc={pxc} ({type(pxc)}), pyc={pyc} ({type(pyc)})')

                roi = vis_frame[py1:py2, px1:px2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                mean = gray.mean()
                std = gray.std()

                highlight_pixels = np.sum(gray > 240)
                highlight_ratio = highlight_pixels / gray.size
                edge_ratio = np.mean(cv2.Canny(gray, 80, 150) > 0)

                if highlight_ratio > 0.003 and std > 40:
                    material = 'silver'
                else:
                    material = 'black'

                self.get_logger().info(
                    f'Mean: {mean}, std: {std}, px: {highlight_pixels}, hratio: {highlight_ratio}, edge: {edge_ratio}, size: {gray.size}'
                )

                detection = Detection2D()
                detection.header = detection_msg.header

                detection.bbox.center.position.x = pxc
                detection.bbox.center.position.y = pyc
                detection.bbox.size_x = float(px2 - px1)
                detection.bbox.size_y = float(py2 - py1)

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = material
                hypothesis.hypothesis.score = float(score)

                detection.results.append(hypothesis)
                detection_msg.detections.append(detection)  # To be sent to ml_rescue_node

                if self.debug:
                    pxc, pyc, px1, py1, px2, py2 = map(int, (pxc, pyc, px1, py1, px2, py2))

                    cv2.rectangle(vis_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                    cv2.circle(vis_frame, (pxc, pyc), 2, (0, 0, 255), -1)
                    # self.get_logger().info(
                    #     f'Object detected at ({pxc},{pyc}) with confidence {score}'
                    # )

                    label = f'{self.model_classes[0]} {score:.2f}'  # Label with confidence score
                    cv2.putText(
                        vis_frame,
                        label,
                        (px1, max(20, py1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

            if latest_balls is None:
                self.get_logger().info('no balls detected')

            green, red = self._drop_point_contours(raw_frame, vis_frame)

            # For context: [colour, xc, yc, width, height]
            for i in [green, red]:
                if len(i) > 0:
                    detection = Detection2D()
                    detection.header = detection_msg.header

                    detection.bbox.center.position.x = float(i[1])
                    detection.bbox.center.position.y = float(i[2])
                    detection.bbox.size_x = float(i[3])
                    detection.bbox.size_y = float(i[4])

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = i[0]

                    detection.results.append(hypothesis)
                    detection_msg.detections.append(detection)

            # Render frame to video
            if self.debug:
                self.out.write(vis_frame)

        except queue.Empty:
            pass

        if detection_msg is not None:
            # self.get_logger().info('- - - Publishing detections - - -')
            self.inference_pub.publish(detection_msg)

        # time_elapsed = time.time() - start_time
        # inference_fps = 1 / time_elapsed
        # if self.debug:
        #     self.get_logger().info(f'--- Fps: {inference_fps:.2f} ---')

    def _drop_point_contours(self, raw_frame, vis_frame):
        """Hectic sketchy temporary evac point finder."""
        green_return = []
        red_return = []

        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        lower_red1 = np.array([0, 200, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 200, 100])
        upper_red2 = np.array([180, 255, 255])
        min_tray_size = 10000

        evac_image = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        evac_image = cv2.GaussianBlur(evac_image, (5, 5), 0)

        green_evac_image = cv2.inRange(evac_image, lower_green, upper_green)
        g_contours, g_heirarchy = cv2.findContours(
            green_evac_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in g_contours:
            # self.get_logger().info(f'Contour with area {cv2.contourArea(contour)}')
            if cv2.contourArea(contour) > min_tray_size:
                gx, gy, gw, gh = cv2.boundingRect(contour)
                gxc = int(gx + (gw / 2))
                gyc = int(gy + (gh / 2))
                cv2.rectangle(vis_frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 3)
                cv2.circle(vis_frame, (gxc, gyc), 2, (0, 0, 255), -1)

                green_return = ['green', gxc, gyc, gw, gh]

            # cv2.drawContours(vis_frame, g_contours, -1, (255, 0, 0), 3)

        red_evac_image1 = cv2.inRange(evac_image, lower_red1, upper_red1)
        red_evac_image2 = cv2.inRange(evac_image, lower_red2, upper_red2)
        red_evac = cv2.bitwise_or(red_evac_image1, red_evac_image2)
        r_contours, r_heirarchy = cv2.findContours(
            red_evac, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in r_contours:
            # self.get_logger().info(f'Contour with area {cv2.contourArea(contour)}')
            if cv2.contourArea(contour) > min_tray_size:
                rx, ry, rw, rh = cv2.boundingRect(contour)
                rxc = int(rx + (rw / 2))
                ryc = int(ry + (rh / 2))
                cv2.rectangle(vis_frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 3)
                cv2.circle(vis_frame, (rxc, ryc), 2, (0, 0, 255), -1)

                red_return = ['red', rxc, ryc, rw, rh]

            # cv2.drawContours(vis_frame, r_contours, -1, (255, 0, 0), 3)

        return [green_return, red_return]

    def _inference_callback(self, completion_info, output_buffer=None, display_frame=None):

        flat_buffer = output_buffer.flatten()
        num_detections = int(flat_buffer[0])
        detections = []

        # self.get_logger().info(f'Flat buffer: {flat_buffer[0]}')

        for i in range(num_detections):
            start_idx = 1 + (i * 5)
            y1 = output_buffer[start_idx]
            x1 = output_buffer[start_idx + 1]
            y2 = output_buffer[start_idx + 2]
            x2 = output_buffer[start_idx + 3]
            score = output_buffer[start_idx + 4]

            if score >= self.conf_threshold:
                detections.append({'box': [y1, x1, y2, x2], 'score': score})

        # self.get_logger().info(f'Completion info: {completion_info}')

        # Push both the frame and its matching detections to the main thread
        if not self.results_queue.full():
            self.results_queue.put_nowait((display_frame, detections))

        # self.inferenceBusy = False


def main(args=None):
    try:
        rclpy.init(args=args)
        vision_node = VisionNode()
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    finally:
        vision_node.out.release()
        vision_node.target.release()
        vision_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
