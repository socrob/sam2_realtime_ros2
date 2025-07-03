
import cv2
import numpy as np
from typing import Union, Tuple, Optional

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
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster


from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from rclpy.clock import Clock

from shapely.geometry import Polygon
from shapely.algorithms.polylabel import polylabel

from sam2_realtime_msgs.msg import TrackedObject

from sam2_realtime.ekf import EKF

class TrackNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("track_node")

        # Parameters        
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        self.declare_parameter('cam_info', '/k4a/rgb/camera_info')
        self.declare_parameter("target_frame", "rgb_camera_link")
        self.declare_parameter("depth_image_units_divisor", 1)
        self.declare_parameter("depth_filter_percentage", 0.3)
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.declare_parameter("min_mask_area", 1000)
        self.declare_parameter('sam2_mask_topic', '/sam2/mask')
        self.declare_parameter("depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("camera_frame", "rgb_camera_link")
        self.declare_parameter("predict_rate", 10)
        self.declare_parameter("print_measurement_marker", True)

        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()
        self.ekf = None


    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")
        
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.cam_info = self.get_parameter('cam_info').get_parameter_value().string_value
        self.sam2_mask_topic = self.get_parameter('sam2_mask_topic').get_parameter_value().string_value
        self.target_frame = (self.get_parameter("target_frame").get_parameter_value().string_value)
        self.camera_frame = (self.get_parameter("camera_frame").get_parameter_value().string_value)
        self.depth_filter_percentage = (self.get_parameter("depth_filter_percentage").get_parameter_value().double_value)
        self.maximum_detection_threshold = (self.get_parameter("maximum_detection_threshold").get_parameter_value().double_value)
        self.min_mask_area = (self.get_parameter("min_mask_area").get_parameter_value().integer_value)
        self.rate = (self.get_parameter("predict_rate").get_parameter_value().integer_value)
        self.depth_image_units_divisor = (self.get_parameter("depth_image_units_divisor").get_parameter_value().integer_value)
        dimg_reliability = (self.get_parameter("depth_image_reliability").get_parameter_value().integer_value)
        dinfo_reliability = (self.get_parameter("depth_info_reliability").get_parameter_value().integer_value)
        self.print_measurement_marker = (self.get_parameter("print_measurement_marker").get_parameter_value().bool_value)

        self.depth_image_qos_profile = QoSProfile(
            reliability=dimg_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )


        self.depth_info_qos_profile = QoSProfile(
            reliability=dinfo_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # TFs
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Position Coordinates
        # x, y, z
        self.position = (0.0, 0.0, 0.0)
        # cx, cy
        self.centroid = (0.0, 0.0)

        # Tracker Publisher
        self._pub = self.create_publisher(TrackedObject, "tracked_object", 10)

        # Debug publisher
        self._meas_marker_pub = self.create_publisher(Marker, "measurement_marker", 10)

        #### EKF Params
        self.initial_state = [0, 0, 1.5, 0, 0, 0]
        
        # Define process noise covariance (Q)
        # Initial state [x, y, z, vx, vy, vz]
        self.initial_state = np.zeros(6)  # [0, 0, 0, 0, 0, 0]
        
        # Define process noise covariance (Q)
        self.process_noise_cov = np.diag([1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3])

        # Azure Kinect depth standard deviation: 5mm = 0.005 meters
        depth_std_dev = 0.005
        measurement_variance = depth_std_dev ** 2

        # Define measurement noise covariance (R) for 3D measurements
        self.measurement_noise_cov = np.eye(3) * measurement_variance  # 3D position measurement noise

        # Initial covariance matrix for state estimation
        self.initial_covariance = np.diag([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        # self.initial_covariance = np.eye(6) * 1e-1  # Initial uncertainty in the state

        # Initialize EKF
        self.ekf = EKF(process_noise_cov=self.process_noise_cov,
                        initial_state=self.initial_state,
                        initial_covariance=self.initial_covariance,
                        dt=1.0/self.rate)
        
        # Debug marker counter
        self.marker_id = 0
        
        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # Subscribers
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=self.depth_image_qos_profile
        )
        self.cam_info_sub = message_filters.Subscriber(
            self, CameraInfo, self.cam_info, qos_profile=self.depth_info_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, TrackedObject, self.sam2_mask_topic
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.cam_info_sub, self.detections_sub), 10, 0.05
        )
        # self._synchronizer = message_filters.TimeSynchronizer(
        #     (self.depth_sub, self.cam_info_sub, self.detections_sub), 10
        # )
        self._synchronizer.registerCallback(self.process_detections)

        self.timer = self.create_timer(1.0 / self.rate, self.run)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn: 
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.depth_sub.sub)
        self.destroy_subscription(self.cam_info_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer
        
        if hasattr(self, 'timer'):
            self.timer.cancel()
            del self.timer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        del self.tf_listener
        del self.tf_broadcaster
        del self.ekf

        self.destroy_publisher(self._pub)
        self.destroy_publisher(self._meas_marker_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS
    

    def run(self) -> None:
        """
        Periodically predicts EKF state and publishes the tracked object message and TF.
        """

        # Transform point
        transform = self.get_transform(self.camera_frame)
        if transform is None:
            self.get_logger().warn(f"[track_node] Could get transform from {self.camera_frame}")
            return
        
        # Get EKF state, i.e. current position
        self.position = tuple(self.ekf.get_state()[:3])
        
        # Apply transform to point
        self.position = TrackNode.transform_point(self.position, transform[0], transform[1])
        
        # Publish msg and TF
        self.publishMessage()

        # Predict
        self.ekf.predict()
    

    def publishMessage(self) -> None:
        tracker_msg = TrackedObject()
        now = Clock().now().to_msg()
        tracker_msg.header.stamp = now
        tracker_msg.header.frame_id = self.target_frame
        tracker_msg.id = 1
        
        tracker_msg.mask.header.stamp = now
        tracker_msg.mask.header.frame_id = self.target_frame

        tracker_msg.position.x = self.position[0]
        # tracker_msg.position.y = self.position[1]
        # Ignoring centroid height which irrelevant for tracking
        tracker_msg.position.y = 0.0
        tracker_msg.position.z = self.position[2]

        tracker_msg.centroid_x = float(self.centroid[0])
        tracker_msg.centroid_y = float(self.centroid[1])

        # self.get_logger().info(f"[track_node] 3D position: x={tracker_msg.position.x:.2f}, y={tracker_msg.position.y:.2f}, z={tracker_msg.position.z:.2f}")

        # Publish TF
        self.publish_tf_from_tracked_object(tracker_msg=tracker_msg)

        self._pub.publish(tracker_msg)

   
    def process_detections(self, depth_msg: Image, cam_info_msg: CameraInfo, tracker_msg: TrackedObject) -> None:
        """
        Processes SAM2 mask and depth data to estimate 3D position of the tracked object.

        Args:
            depth_msg (Image): Depth image.
            cam_info_msg (CameraInfo): Camera intrinsic matrix info.
            tracker_msg (TrackedObject): Output message to populate.

        Returns:
            TrackedObject: Updated message with 3D position filled in.
        """
        # If mask is not available, there is no point in tracking
        if not tracker_msg.mask:
            return
        
        # Convert imgs
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        mask = self.cv_bridge.imgmsg_to_cv2(tracker_msg.mask, desired_encoding="mono8")

        # Check mask size
        num_pixels = cv2.countNonZero(mask)
        if num_pixels < self.min_mask_area:
            self.get_logger().warn(f"[track_node] Mask too small ({num_pixels} pixels) — ignoring this measurement.")
            return

        # Try using centroid of the mask first
        cx, cy = self.get_centroid_of_mask(mask)
        if cx == -1 or cy == -1 or mask[int(cy), int(cx)] != 255:
            cx, cy = self.get_furthest_point_from_mask_edge(mask)
            if cx == -1 or cy == -1:
                self.get_logger().warn('[track_node] Could not compute a valid point in the mask')
                return
            
        # Estimate depth of a square centered in cx, cy
        # depth = self.get_median_depth(int(cy), int(cx), depth_image, mask)
        depth = self.get_median_depth_2(depth_image, mask)
        
        if depth <= 0:
            return

        # Convert depth image coordinates to 3D camera space
        k = cam_info_msg.k
        fx, fy, px, py = k[0], k[4], k[2], k[5]

        z = depth
        x = (int(cx) - px) * z / fx
        y = (int(cy) - py) * z / fy

        if self.print_measurement_marker:
            self.debug_marker(x=float(x), y=0.0, z=float(z))

        # Update EKF after measurement
        self.ekf.update([x, y, z], dynamic_R=self.measurement_noise_cov)

        self.centroid = (float(cx), float(cy))

        return
    

    def debug_marker(self, x: float, y: float, z:float):
        # --- Debug marker for measurement ---
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.camera_frame

        marker.ns = "measurement"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.05  # 5 cm sphere
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self._meas_marker_pub.publish(marker)

    

    def publish_tf_from_tracked_object(self, tracker_msg: TrackedObject):
        """
        Publishes a TF transform from the tracked object's position.

        Args:
            tracker_msg (TrackedObject): Message containing position and id.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = tracker_msg.header.frame_id
        t.child_frame_id = f"tracked_object_{tracker_msg.id}"

        t.transform.translation.x = tracker_msg.position.x
        t.transform.translation.y = tracker_msg.position.y
        t.transform.translation.z = tracker_msg.position.z

        # Identity rotation
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
    

    def get_median_depth(self, cy: int, cx: int, depth_image: np.ndarray, mask: Optional[np.ndarray]) -> float:
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
            median = float(median/self.depth_image_units_divisor)
            return 0.0 if np.isnan(median) else median
        return 0.0
    

    def get_median_depth_2(self, depth_image: np.ndarray, mask: Optional[np.ndarray]) -> float:
        """

        """
        if mask is None or mask.shape[:2] != depth_image.shape[:2]:
            return 0.0

        # Apply the mask directly
        roi = cv2.bitwise_and(depth_image, depth_image, mask=mask)

        # Convert to meters (camera dependent)
        roi = roi / self.depth_image_units_divisor
        if not np.any(roi):
            return 0.0
        
        # Compute the median Z value of the object from the mask
        roi = roi[roi > 0]
        bb_center_z_coord = np.median(roi)

        # This computes the absolute difference between each depth value in the ROI and the estimated center Z value of the bounding box 
        # (matrix of how far each point in ROI is from the center depth, in meters)
        z_diff = np.abs(roi - bb_center_z_coord)
        # A binary mask that selects only the pixels in ROI where the depth is within a small threshold of the center depth.
        mask_z = z_diff <= self.maximum_detection_threshold
        if not np.any(mask_z):
            return 0.0

        # Now roi is a 1D array of filtered depth values close to the center Z
        roi = roi[mask_z]
        # Compute the range of depth values within that filtered zone
        z_min, z_max = np.min(roi), np.max(roi)
        # Compute the average depth within that filtered zone
        z = float((z_max + z_min) / 2)

        return z if z > 0 else 0.0


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
    
    @staticmethod
    def transform_point(
        position: Tuple,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> Tuple:

        # Rotate the position vector
        rotated_position = TrackNode.qv_mult(
            rotation,
            np.array([
                position[0],
                position[1],
                position[2],
            ])
        )

        # Apply translation
        position = rotated_position + translation

        return position

    

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (uv * q[0] + uuv)


def main():
    rclpy.init()
    node = TrackNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
