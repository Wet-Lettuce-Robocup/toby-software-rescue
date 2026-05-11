import rclpy# pyright: ignore[reportMissingImports]
from rclpy.node import Node# pyright: ignore[reportMissingImports]
from std_msgs.msg import String# pyright: ignore[reportMissingImports]
from geometry_msgs.msg import Twist # pyright: ignore[reportMissingImports]
from rclpy.lifecycle import LifecycleNode, LifecycleState # pyright: ignore[reportMissingImports]

class TRescue(LifecycleNode):
    def __init__(self) -> None:
        super().__init__('ml_rescue')
        
class TestPublisher(Node):
    def __init__(self) -> None:
        super().__init__('test_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(0.5, self.publish_message)
        self.i=0

    def publish_message(self):
        msg = String()
        msg.data = 'Hello world! %d' % self.i
        
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = TestPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':    main()