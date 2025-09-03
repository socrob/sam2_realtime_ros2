import cv2
import numpy as np
from typing import Union, Tuple, Optional, List

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.time import Time

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from rclpy.duration import Duration
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pcl2
from geometry_msgs.msg import TransformStamped

from shapely.geometry import Polygon
from shapely.algorithms.polylabel import polylabel

from sam2_realtime_msgs.msg import TrackedObject


class TrackNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("mask2pcl")

        # Parameters
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        self.declare_parameter('cam_info', '/k4a/rgb/camera_info')
        self.declare_parameter("target_frame", "camera_base")
        self.declare_parameter("depth_image_units_divisor", 1)  # e.g., 1000 for mm→m
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.declare_parameter('sam2_mask_topic', '/sam2/mask')
        self.declare_parameter("depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("tracking_active", False)  # event_in
        self.declare_parameter("min_mask_area", 100)       # reject tiny masks
        self.declare_parameter("cloud_stride", 4)          # downsample factor when building PCL

        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()

    # ---------------- Lifecycle -----------------
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.cam_info = self.get_parameter('cam_info').get_parameter_value().string_value
        self.sam2_mask_topic = self.get_parameter('sam2_mask_topic').get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.maximum_detection_threshold = self.get_parameter("maximum_detection_threshold").get_parameter_value().double_value
        self.depth_image_units_divisor = self.get_parameter("depth_image_units_divisor").get_parameter_value().integer_value
        dimg_reliability = self.get_parameter("depth_image_reliability").get_parameter_value().integer_value
        self.tracking_active = self.get_parameter("tracking_active").get_parameter_value().bool_value
        self.min_mask_area = self.get_parameter("min_mask_area").get_parameter_value().integer_value
        self.cloud_stride = max(1, self.get_parameter("cloud_stride").get_parameter_value().integer_value)

        self.depth_image_qos_profile = QoSProfile(
            reliability=dimg_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        dinfo_reliability = self.get_parameter("depth_info_reliability").get_parameter_value().integer_value
        self.depth_info_qos_profile = QoSProfile(
            reliability=dinfo_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Publishers
        self._pcl_pub = self.create_publisher(PointCloud2, "object_cloud", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # Subscribers
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=self.depth_image_qos_profile)
        self.cam_info_sub = message_filters.Subscriber(self, CameraInfo, self.cam_info, qos_profile=self.depth_info_qos_profile)
        self.detections_sub = message_filters.Subscriber(self, TrackedObject, self.sam2_mask_topic)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.cam_info_sub, self.detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.on_detections)

        self._event_sub = self.create_subscription(String, "event_in", self.event_callback, 10)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")
        self.destroy_subscription(self.depth_sub.sub)
        self.destroy_subscription(self.cam_info_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)
        del self._synchronizer
        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        self.destroy_publisher(self._pcl_pub)
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    # ---------------- Callbacks -----------------
    def on_detections(self, depth_msg: Image, cam_info_msg: CameraInfo, detections_msg: TrackedObject) -> None:
        if not self.tracking_active:
            return

        # Just forward the depth, camera info, and the mask image to processing
        mask_msg = detections_msg.mask
        # If the mask header frame is missing, fall back to the depth frame (not strictly required here,
        # but keeps things consistent if you log/debug headers later)
        if not mask_msg.header.frame_id:
            mask_msg.header.frame_id = depth_msg.header.frame_id

        pcl_msg = self.process_detections(depth_msg, cam_info_msg, mask_msg)
        if pcl_msg is not None:
            self._pcl_pub.publish(pcl_msg)


    # --------------- Core Logic -----------------
    def process_detections(self, depth_msg: Image, cam_info_msg: CameraInfo, mask_msg: Image) -> Optional[PointCloud2]:
        """
        Build a PointCloud2 from the object's mask and depth, and transform to target frame.
        """
        # If mask is not available, nothing to do
        if mask_msg is None:
            return None

        # Convert images
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        mask = self.cv_bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")

        # Check mask size (ignore if mask is too small)
        num_pixels = cv2.countNonZero(mask)
        if num_pixels < self.min_mask_area:
            self.get_logger().warn(f"[mask2pcl] Mask too small ({num_pixels} pixels) — ignoring this measurement.")
            return None

        # Camera intrinsics
        k = cam_info_msg.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]

        # Build 3D points (in depth frame)
        src_frame = depth_msg.header.frame_id
        stamp = depth_msg.header.stamp
        points_cam = self.mask_to_points(depth_image, mask, fx, fy, cx, cy)
        if len(points_cam) == 0:
            return None

        # Transform points to target_frame if needed
        tgt_frame = self.target_frame
        if src_frame != tgt_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    tgt_frame, src_frame, Time.from_msg(stamp), Duration(seconds=0.2)
                )
                points_cam = self.apply_transform_to_points(points_cam, tf)
                header_frame = tgt_frame
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed {src_frame}→{tgt_frame}: {ex}. Publishing in source frame.")
                header_frame = src_frame
        else:
            header_frame = src_frame

        # Create PointCloud2 with a fresh header (don’t mutate incoming message headers)
        header = Header()
        header.frame_id = header_frame
        header.stamp = stamp

        pcl_msg = pcl2.create_cloud_xyz32(header, points_cam)
        return pcl_msg


    # ------------ Utilities ---------------------
    def mask_to_points(self, depth: np.ndarray, mask: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> List[Tuple[float, float, float]]:
        """Convert a masked depth image into a list of 3D points (meters) using pinhole model.
        Downsamples by self.cloud_stride for performance.
        """
        if depth.shape[:2] != mask.shape[:2]:
            self.get_logger().warn("Depth and mask shapes differ; skipping.")
            return []

        stride = self.cloud_stride
        mask_bool = mask > 0
        ys, xs = np.where(mask_bool)
        if ys.size == 0:
            return []

        # Downsample indices
        ys = ys[::stride]
        xs = xs[::stride]

        z = depth[ys, xs].astype(np.float32) / max(1, self.depth_image_units_divisor)
        valid = z > 0
        if not np.any(valid):
            return []
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        z = z[valid]

        # Project to 3D (camera/depth frame)
        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        points = np.stack((X, Y, z), axis=-1)
        return points.tolist()

    def apply_transform_to_points(self, points: List[Tuple[float, float, float]], tf: TransformStamped) -> List[Tuple[float, float, float]]:
        """Apply a geometry_msgs/TransformStamped to a list of (x,y,z) points using do_transform_point.
        This is simple and safe, and with stride it's fast enough in Python.
        """
        out: List[Tuple[float, float, float]] = []
        ps = PointStamped()
        ps.header.stamp = tf.header.stamp
        ps.header.frame_id = tf.child_frame_id  # source frame
        for x, y, z in points:
            ps.point.x, ps.point.y, ps.point.z = float(x), float(y), float(z)
            tp = do_transform_point(ps, tf)
            out.append((tp.point.x, tp.point.y, tp.point.z))
        return out

    def get_median_depth_2(self, depth_image: np.ndarray, mask: Optional[np.ndarray]) -> float:
        if mask is None or mask.shape[:2] != depth_image.shape[:2]:
            return 0.0
        roi = cv2.bitwise_and(depth_image, depth_image, mask=mask)
        roi = roi / max(1, self.depth_image_units_divisor)
        if not np.any(roi):
            return 0.0
        roi = roi[roi > 0]
        bb_center_z_coord = np.median(roi)
        z_diff = np.abs(roi - bb_center_z_coord)
        mask_z = z_diff <= self.maximum_detection_threshold
        if not np.any(mask_z):
            return 0.0
        roi = roi[mask_z]
        z_min, z_max = np.min(roi), np.max(roi)
        z = float((z_max + z_min) / 2)
        return z if z > 0 else 0.0

    def get_centroid_of_mask(self, mask: np.ndarray) -> Tuple[Union[int, float], Union[int, float]]:
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return -1, -1
        largest_contour = max(contours, key=cv2.contourArea)
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
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        if np.count_nonzero(mask_gray) == 0:
            self.get_logger().warn("Received an empty mask. Cannot compute furthest point.")
            return -1, -1
        distance_transform = cv2.distanceTransform(mask_gray, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
        cy, cx = np.where(distance_transform == distance_transform.max())
        return int(cx[0]), int(cy[0])

    # ---------------- Events --------------------
    def event_callback(self, msg: String):
        if msg.data == "e_stop":
            self.tracking_active = False
            self.get_logger().info("[track_node] Received e_stop → pausing tracking.")
        elif msg.data == "e_start":
            self.tracking_active = True
            self.get_logger().info("[track_node] Received e_start → resuming tracking.")
        else:
            self.get_logger().warn(f"[track_node] Unknown event: '{msg.data}'")


def main():
    rclpy.init()
    node = TrackNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
