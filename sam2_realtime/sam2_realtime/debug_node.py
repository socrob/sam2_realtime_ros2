# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import numpy as np

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


from sam2_realtime_msgs.msg import TrackedObject


class Sam2DebugNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("sam2_debug_node")

        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('image_topic', '/k4a/rgb/image_raw')
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter('tracker_topic', '/sam2/mask')

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.tracker_topic = self.get_parameter('tracker_topic').get_parameter_value().string_value

        self.image_qos_profile = QoSProfile(
            reliability=self.get_parameter("image_reliability")
            .get_parameter_value()
            .integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # pubs
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._bb_markers_pub = self.create_publisher(MarkerArray, "dgb_bb_markers", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # subs
        self.image_sub = message_filters.Subscriber(self, Image, self.image_topic, qos_profile=self.image_qos_profile)
        self.tracker_sub = message_filters.Subscriber(self, TrackedObject, self.tracker_topic, qos_profile=10)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer((self.image_sub, self.tracker_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.tracker_cb)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.image_sub.sub)
        self.destroy_subscription(self.tracker_sub.sub)

        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._dbg_pub)
        self.destroy_publisher(self._bb_markers_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS


    def tracker_cb(self, img_msg: Image, tracker_msg: TrackedObject) -> None:
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            mask = self.cv_bridge.imgmsg_to_cv2(tracker_msg.mask, desired_encoding="mono8")

            # Create a color mask
            color_mask = np.zeros_like(cv_image)
            color_mask[mask == 255] = (0, 255, 0)

            # Overlay mask on image
            overlay = cv2.addWeighted(cv_image, 1.0, color_mask, 0.5, 0)

            # Draw centroid
            cx, cy = int(tracker_msg.centroid_x), int(tracker_msg.centroid_y)
            if cx >= 0 and cy >= 0:
                cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(overlay, f"ID: {tracker_msg.id}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            debug_msg = self.cv_bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            debug_msg.header = img_msg.header
            self._dbg_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().warn(f"Failed to render debug image: {e}")

        

def main():
    rclpy.init()
    node = Sam2DebugNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
