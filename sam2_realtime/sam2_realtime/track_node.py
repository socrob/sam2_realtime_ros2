#!/usr/bin/env python3

import traceback
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped

from sam2_realtime_msgs.msg import TrackedObject

from cv_bridge import CvBridge
import message_filters
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

import numpy as np
from typing import Tuple, Union, Optional
import cv2

from shapely.geometry import Polygon
from shapely.algorithms import polylabel

class TrackNode(LifecycleNode):

    def __init__(self):
        super().__init__('track_node')

        # Declare parameters
        self.declare_parameter('depth_topic', '/camera/camera/extrinsics/depth_to_color')
        self.declare_parameter('depth_info', '/camera/camera/depth/camera_info')
        self.declare_parameter('sam2_mask_topic', '/sam2/mask')

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("depth_filter_percentage", 0.2)
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter("depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[track_node] Configuring...')

            # Load parameters
            self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
            self.depth_info = self.get_parameter('depth_info').get_parameter_value().string_value
            self.sam2_mask_topic = self.get_parameter('sam2_mask_topic').get_parameter_value().string_value

            self.target_frame = (self.get_parameter("target_frame").get_parameter_value().string_value)
            self.depth_filter_percentage = (self.get_parameter("depth_filter_percentage").get_parameter_value().double_value)
            self.depth_image_units_divisor = (self.get_parameter("depth_image_units_divisor").get_parameter_value().integer_value)
            dimg_reliability = (self.get_parameter("depth_image_reliability").get_parameter_value().integer_value)

            self.depth_image_qos_profile = QoSProfile(
                reliability=dimg_reliability,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )

            dinfo_reliability = (self.get_parameter("depth_info_reliability").get_parameter_value().integer_value)

            self.depth_info_qos_profile = QoSProfile(
                reliability=dinfo_reliability,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            self.tf_listener = TransformListener(self.tf_buffer, self)

            # Publisher
            self._pub = self.create_publisher(TrackedObject, "tracked_object", 10)

            super().on_configure(state)
            self.get_logger().info('[track_node] Configured')

            return TransitionCallbackReturn.SUCCESS
        
        except Exception as e:
            self.get_logger().error(f"[track_node] Exception during configuration: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[track_node] Activating...')

            # Subscribers
            self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=self.depth_image_qos_profile)
            self.depth_info_sub = message_filters.Subscriber(self, CameraInfo, self.depth_info, qos_profile=self.depth_info_qos_profile)
            self.detections_sub = message_filters.Subscriber(self, TrackedObject, self.sam2_mask_topic)

            self._synchronizer = message_filters.ApproximateTimeSynchronizer((self.depth_sub, self.depth_info_sub, self.detections_sub), 10, 0.5)
            self._synchronizer.registerCallback(self.on_detections)

            super().on_activate(state)
            self.get_logger().info('[track_node] Subscriptions activated')
            return TransitionCallbackReturn.SUCCESS
        
        except Exception as e:
            self.get_logger().error(f"[track_node] Exception during activation: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[track_node] Deactivating...')
            self.destroy_subscription(self.depth_sub.sub)
            self.destroy_subscription(self.depth_info_sub.sub)
            self.destroy_subscription(self.detections_sub.sub)
            del self._synchronizer
            super().on_deactivate(state)
            self.get_logger().info('[track_node] Deactivated')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[track_node] Exception during deactivation: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[track_node] Cleaning up...')
            del self.tf_listener
            self.destroy_publisher(self._pub)
            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[track_node] Exception during cleanup: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[track_node] Shutting down...')
            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[track_node] Exception during shutdown: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE


    def on_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: TrackedObject,
    ) -> None:

        new_detections_msg = TrackedObject()
        new_detections_msg.header = detections_msg.header
        new_detections_msg = self.process_detections(
            depth_msg, depth_info_msg, detections_msg, new_detections_msg
        )
        self._pub.publish(new_detections_msg)
        # Publish TF
    

    def process_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: TrackedObject,
        new_detections_msg: TrackedObject,
    ) -> TrackedObject:
        """
        Processes SAM2 mask and depth data to estimate 3D position of the tracked object.

        Args:
            depth_msg (Image): Depth image.
            depth_info_msg (CameraInfo): Camera intrinsic matrix info.
            detections_msg (TrackedObject): Input message with mask and header.
            new_detections_msg (TrackedObject): Output message to populate.

        Returns:
            TrackedObject: Updated message with 3D position filled in.
        """
        if not detections_msg.mask:
            return new_detections_msg

        transform = self.get_transform(depth_info_msg.header.frame_id)
        if transform is None:
            return new_detections_msg

        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        mask = self.cv_bridge.imgmsg_to_cv2(detections_msg.mask, desired_encoding="mono8")

        # Try using centroid of the mask first
        cx, cy = self.get_centroid_of_mask(mask)
        if cx == -1 or cy == -1 or mask[int(cy), int(cx), 0] != 255:
            cx, cy = self.get_furthest_point_from_mask_edge(mask)
            if cx == -1 or cy == -1:
                self.get_logger().warn('[track_node] Could not compute a valid point in the mask')
                return new_detections_msg

        # Estimate depth
        depth = self.get_median_depth(int(cy), int(cx), depth_image, mask)
        if depth <= 0:
            return new_detections_msg

        # Convert depth image coordinates to 3D camera space
        k = depth_info_msg.k
        fx, fy, px, py = k[0], k[4], k[2], k[5]

        z = depth / self.depth_image_units_divisor
        x = (int(cx) - px) * z / fx
        y = (int(cy) - py) * z / fy

        # TODO: Apply Kalman Filter here (if needed)
        # TODO: Optionally apply frame transformation here

        self.get_logger().info(f"[track_node] 3D position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

        # Populate output message
        new_detections_msg.position.x = x
        new_detections_msg.position.y = y
        new_detections_msg.position.z = z

        return new_detections_msg


    def get_median_depth(
        self,
        cy: int,
        cx: int,
        depth_image: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> float:
        """
        Calculates the median depth within a rectangular region around (cy, cx),
        optionally filtering with a binary mask.

        Args:
            cy (int): Y-coordinate of center point.
            cx (int): X-coordinate of center point.
            depth_image (np.ndarray): Depth image (2D array).
            mask (Optional[np.ndarray]): Binary mask to filter depth pixels (same size as depth_image).

        Returns:
            float: Median depth value or 0.0 if invalid.
        """
        height, width = depth_image.shape[:2]

        half_height = int(height * self.depth_filter_percentage)
        half_width = int(width * self.depth_filter_percentage)

        y0 = max(0, cy - half_height)
        y1 = min(height, cy + half_height + 1)
        x0 = max(0, cx - half_width)
        x1 = min(width, cx + half_width + 1)

        depth_region = depth_image[y0:y1, x0:x1]

        if mask is not None:
            mask_region = mask[y0:y1, x0:x1]
            if mask_region.ndim == 3:  # If mask is RGB, convert to grayscale
                mask_region = cv2.cvtColor(mask_region, cv2.COLOR_BGR2GRAY)
            valid_depths = depth_region[mask_region == 255]
        else:
            valid_depths = depth_region.flatten()

        if valid_depths.size > 0:
            median = np.median(valid_depths)
            return 0.0 if np.isnan(median) else float(median)
        return 0.0



    def get_centroid_of_mask(self, mask: np.ndarray) -> Tuple[Union[int, float], Union[int, float]]:
        """
        Computes the centroid of the largest contour in the mask using polylabel.

        Args:
            mask (np.ndarray): Binary or BGR mask image.

        Returns:
            Tuple[int | float, int | float]: (cx, cy) coordinates of the centroid, or (-1, -1) if failed.
        """
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return -1, -1

        largest_contour = max(contours, key=cv2.contourArea)

        # Handle cases where contour is too small or empty
        if largest_contour.shape[0] < 3:
            return -1, -1

        try:
            polygon = Polygon(largest_contour.squeeze())
            if not polygon.is_valid or polygon.is_empty:
                return -1, -1

            centroid = polylabel(polygon, tolerance=1.0)
            return centroid.x, centroid.y
        except Exception as e:
            self.get_logger().warn(f"Polylabel centroid calculation failed: {e}")
            return -1, -1


    def get_furthest_point_from_mask_edge(self, mask: np.ndarray) -> Tuple[int, int]:
        """
        Finds the point within the mask that is furthest from the edge using a distance transform.

        Args:
            mask (np.ndarray): Binary or grayscale mask image (BGR accepted, auto-converted).

        Returns:
            Tuple[int, int]: Coordinates (x, y) of the furthest point from the edge.
        """
        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        # Check if the mask is valid (non-zero)
        if np.count_nonzero(mask_gray) == 0:
            self.get_logger().warn("Received an empty mask. Cannot compute furthest point.")
            return -1, -1

        distance_transform = cv2.distanceTransform(mask_gray, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
        cy, cx = np.where(distance_transform == distance_transform.max())

        return int(cx[0]), int(cy[0])



    def get_transform(self, frame_id: str) -> Tuple[np.ndarray]:
        # transform position from image frame to target_frame
        rotation = None
        translation = None

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, rclpy.time.Time()
            )

            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )

            rotation = np.array(
                [
                    transform.transform.rotation.w,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                ]
            )

            return translation, rotation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None


def main():
    rclpy.init()
    node = TrackNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
