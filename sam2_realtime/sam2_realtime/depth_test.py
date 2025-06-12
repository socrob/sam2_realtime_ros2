import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sam2_realtime_msgs.msg import TrackedObject

import message_filters

class SyncTest(Node):
    def __init__(self):
        super().__init__('sync_test')

        self.get_logger().info("Initialized SyncTest node!")

        # Subscribers
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        self.mask_sub = message_filters.Subscriber(self, TrackedObject, '/sam2/mask')

        # Synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.mask_sub],
            queue_size=30,
            slop=0.2
        )
        self.ts.registerCallback(self.synced_callback)

    def synced_callback(self, depth_msg, mask_msg):
        self.get_logger().info(f"SYNCED: depth @ {depth_msg.header.stamp.sec}.{depth_msg.header.stamp.nanosec} | mask @ {mask_msg.header.stamp.sec}.{mask_msg.header.stamp.nanosec}")

def main():
    rclpy.init()
    node = SyncTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
