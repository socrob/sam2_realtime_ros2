
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
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster

from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from std_msgs.msg import String

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
        self.declare_parameter("predict_rate", 10)
        self.declare_parameter("print_measurement_marker", True)
        self.declare_parameter("max_position_jump", 0.3) #meters
        self.declare_parameter("relock_window", 1) #seconds
        self.declare_parameter("enable", False) #event_in

        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("processing_timer_period", 0.001)

        self.declare_parameter("fix_height", False)
        self.declare_parameter("fixed_height", 1.70)

        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()
        self.ekf = None


    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.cam_info = self.get_parameter('cam_info').get_parameter_value().string_value
        self.sam2_mask_topic = self.get_parameter('sam2_mask_topic').get_parameter_value().string_value
        self.target_frame = (self.get_parameter("target_frame").get_parameter_value().string_value)
        self.depth_filter_percentage = (self.get_parameter("depth_filter_percentage").get_parameter_value().double_value)
        self.maximum_detection_threshold = (self.get_parameter("maximum_detection_threshold").get_parameter_value().double_value)
        self.min_mask_area = (self.get_parameter("min_mask_area").get_parameter_value().integer_value)
        self.rate = (self.get_parameter("predict_rate").get_parameter_value().integer_value)
        self.depth_image_units_divisor = (self.get_parameter("depth_image_units_divisor").get_parameter_value().integer_value)
        dimg_reliability = (self.get_parameter("depth_image_reliability").get_parameter_value().integer_value)
        dinfo_reliability = (self.get_parameter("depth_info_reliability").get_parameter_value().integer_value)
        self.print_measurement_marker = (self.get_parameter("print_measurement_marker").get_parameter_value().bool_value)
        self.max_position_jump = (self.get_parameter("max_position_jump").get_parameter_value().double_value)
        self.relock_window = (self.get_parameter("relock_window").get_parameter_value().integer_value)
        self.enable = (self.get_parameter("enable").get_parameter_value().bool_value)
        self.camera_frame = None

        self.sync_queue_size = self.get_parameter("sync_queue_size").get_parameter_value().integer_value
        self.sync_slop = self.get_parameter("sync_slop").get_parameter_value().double_value
        self.processing_timer_period = self.get_parameter("processing_timer_period").get_parameter_value().double_value

        self.fix_height = self.get_parameter("fix_height").get_parameter_value().bool_value
        self.fixed_height = self.get_parameter("fixed_height").get_parameter_value().double_value

        self.camera_frame = None
        self.cam_info_msg = None
        self.latest_packet = None
        self.processing_measurement = False

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
        self.transformed_position = (0.0, 0.0, 0.0)
        # cx, cy
        self.centroid = (0.0, 0.0)

        # Tracker Publisher
        self._pub = self.create_publisher(TrackedObject, "tracked_object", 10)

        # (TODO) ROS1 publisher
        self._point_pub = self.create_publisher(PointStamped, '/ros1_tracked_position', 10)

        # Debug publisher
        self._meas_marker_pub = self.create_publisher(Marker, "measurement_marker", 10)

        #### EKF Params
        self.initial_state = [0, 0, 1.5, 0, 0, 0]

        # Define process noise covariance (Q)
        # Initial state [x, y, z, vx, vy, vz]
        self.initial_state = np.zeros(6)
        if self.fix_height:
            self.initial_state[2] = self.fixed_height

        # Define process noise covariance (Q)
        # self.process_noise_cov = np.diag([1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3])
        self.process_noise_cov = np.diag([
            5e-4, 5e-4, 1e-5,
            1e-4, 1e-4, 1e-6,
        ])

        # Initial covariance matrix for state estimation
        self.initial_covariance = np.diag([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

        # Azure Kinect depth standard deviation: 5mm = 0.005 meters
        # depth_std_dev = 0.005
        # measurement_variance = depth_std_dev ** 2

        # Define measurement noise covariance (R) for 3D measurements
        # self.measurement_noise_cov = np.eye(3) * measurement_variance  # 3D position measurement noise

        measurement_std_dev = 0.12
        measurement_variance = measurement_std_dev ** 2
        self.measurement_noise_cov = np.eye(3) * measurement_variance

        if self.fix_height:
            self.measurement_noise_cov[2, 2] = 1.0

        # Initialize EKF
        self.ekf = EKF(process_noise_cov=self.process_noise_cov,
                        initial_state=self.initial_state,
                        initial_covariance=self.initial_covariance,
                        dt=1.0/self.rate)

        # Debug marker counter
        self.marker_id = 0

        # Time gated outlier rejection params
        self.last_update_time = self.get_clock().now()
        self.has_measurement = False

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        self.cam_info_msg = None
        self.latest_packet = None
        self.processing_measurement = False
        self.has_measurement = False

        # Subscribe once to CameraInfo.
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            self.cam_info,
            self.cam_info_cb,
            self.depth_info_qos_profile,
        )

        # Synchronize only the frame-varying streams.
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=self.depth_image_qos_profile,
        )

        self.detections_sub = message_filters.Subscriber(
            self,
            TrackedObject,
            self.sam2_mask_topic,
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.detections_sub),
            self.sync_queue_size,
            self.sync_slop,
        )
        self._synchronizer.registerCallback(self.synced_measurement_cb)

        self._event_sub = self.create_subscription(
            String,
            "event_in",
            self.event_callback,
            10,
        )

        # Processes latest depth+mask measurement outside the sync callback.
        self.measurement_timer = self.create_timer(
            self.processing_timer_period,
            self.process_latest_measurement,
        )

        # Existing EKF prediction/publishing timer.
        self.timer = self.create_timer(
            1.0 / self.rate,
            self.run,
        )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS


    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.camera_frame = None
        self.cam_info_msg = None
        self.latest_packet = None
        self.processing_measurement = False

        if hasattr(self, "timer") and self.timer is not None:
            self.destroy_timer(self.timer)
            self.timer = None

        if hasattr(self, "measurement_timer") and self.measurement_timer is not None:
            self.destroy_timer(self.measurement_timer)
            self.measurement_timer = None

        if hasattr(self, "depth_sub") and self.depth_sub is not None:
            self.destroy_subscription(self.depth_sub.sub)
            self.depth_sub = None

        if hasattr(self, "detections_sub") and self.detections_sub is not None:
            self.destroy_subscription(self.detections_sub.sub)
            self.detections_sub = None

        if hasattr(self, "cam_info_sub") and self.cam_info_sub is not None:
            self.destroy_subscription(self.cam_info_sub)
            self.cam_info_sub = None

        if hasattr(self, "_event_sub") and self._event_sub is not None:
            self.destroy_subscription(self._event_sub)
            self._event_sub = None

        if hasattr(self, "_synchronizer"):
            del self._synchronizer

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
        # TODO ROS1 fix
        self.destroy_publisher(self._point_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_shutdown(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS


    def cam_info_cb(self, msg: CameraInfo) -> None:
        """
        Store CameraInfo once and unsubscribe.

        CameraInfo is static for a fixed camera configuration, so it does not need
        to be synchronized every frame.
        """
        self.cam_info_msg = msg
        self.camera_frame = msg.header.frame_id

        self.get_logger().info(
            f"[track_node] Received CameraInfo from frame '{msg.header.frame_id}'. "
            "Unsubscribing from cam_info."
        )

        if hasattr(self, "cam_info_sub") and self.cam_info_sub is not None:
            self.destroy_subscription(self.cam_info_sub)
            self.cam_info_sub = None


    def synced_measurement_cb(
        self,
        depth_msg: Image,
        tracker_msg: TrackedObject,
    ) -> None:
        """
        Store the latest synchronized depth + SAM2 mask packet.

        Heavy measurement extraction and EKF update happen in process_latest_measurement().
        """
        if not self.enable:
            return

        self.latest_packet = (depth_msg, tracker_msg)


    def reset_ekf_state(self, position: np.ndarray) -> None:
        self.initial_state = np.zeros(6)
        self.initial_state[0] = float(position[0])
        self.initial_state[1] = float(position[1])
        self.initial_state[2] = float(position[2])

        self.ekf = EKF(
            process_noise_cov=self.process_noise_cov,
            initial_state=self.initial_state,
            initial_covariance=self.initial_covariance,
            dt=1.0 / self.rate,
        )

    def run(self) -> None:
        """
        Periodically predicts EKF state and publishes the tracked object message and TF.

        The EKF state is already expressed in target_frame, because measurements are
        transformed into target_frame before EKF update.
        """
        if not self.enable:
            return

        # Wait until we have received at least one valid measurement.
        if self.camera_frame is None or not self.has_measurement:
            return

        self.position = tuple(self.ekf.get_state()[:3])

        if self.fix_height:
            self.position = (
                self.position[0],
                self.position[1],
                self.fixed_height,
            )

        self.transformed_position = self.position

        self.publishMessage()
        self.ekf.predict()


    def publishMessage(self) -> None:
        tracker_msg = TrackedObject()
        now = Clock().now().to_msg()
        tracker_msg.header.stamp = now
        tracker_msg.header.frame_id = self.target_frame
        tracker_msg.id = 1

        tracker_msg.mask.header.stamp = now
        tracker_msg.mask.header.frame_id = self.target_frame

        tracker_msg.position.x = self.transformed_position[0]
        tracker_msg.position.y = self.transformed_position[1]
        tracker_msg.position.z = self.transformed_position[2]

        tracker_msg.centroid_x = float(self.centroid[0])
        tracker_msg.centroid_y = float(self.centroid[1])

        # self.get_logger().info(f"[track_node] 3D position: x={tracker_msg.position.x:.2f}, y={tracker_msg.position.y:.2f}, z={tracker_msg.position.z:.2f}")

        # Publish TF
        self.publish_tf_from_tracked_object(tracker_msg=tracker_msg)

        self._pub.publish(tracker_msg)

        # Publish PointStamped message
        # (#TODO Remove further ahead. ROS1 fix)
        point_msg = PointStamped()
        point_msg.header.stamp = now
        point_msg.header.frame_id = self.target_frame
        point_msg.point = tracker_msg.position

        self._point_pub.publish(point_msg)


    def process_latest_measurement(self) -> None:
        """
        Processes the latest SAM2 mask and depth data to estimate the 3D position
        of the tracked object, then updates the EKF.

        This runs outside the message_filters callback.
        """
        if self.processing_measurement:
            return

        if not self.enable:
            self.latest_packet = None
            return

        if self.latest_packet is None:
            return

        if self.cam_info_msg is None:
            self.get_logger().warn(
                "[track_node] Waiting for CameraInfo before processing measurements.",
                throttle_duration_sec=2.0,
            )
            return

        depth_msg, tracker_msg = self.latest_packet
        self.latest_packet = None

        self.processing_measurement = True

        try:
            self.process_detections(depth_msg, self.cam_info_msg, tracker_msg)

        except Exception as e:
            self.get_logger().error(f"[track_node] Measurement processing failed: {e}")

        finally:
            self.processing_measurement = False


    def process_detections(
        self,
        depth_msg: Image,
        cam_info_msg: CameraInfo,
        tracker_msg: TrackedObject,
    ) -> None:
        """
        Processes SAM2 mask and depth data to estimate 3D position of the tracked object.

        The raw measurement is first computed in the camera frame, then transformed
        into target_frame. The EKF is updated in target_frame.
        """
        if not self.enable:
            return

        # If mask is not available, there is no point in tracking.
        if not tracker_msg.mask:
            return

        # Reads camera_frame
        self.camera_frame = cam_info_msg.header.frame_id

        # Convert imgs
        depth_image = self.cv_bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough",
        )
        mask = self.cv_bridge.imgmsg_to_cv2(
            tracker_msg.mask,
            desired_encoding="mono8",
        )

        # Check mask size, ignore if mask is too small.
        num_pixels = cv2.countNonZero(mask)
        if num_pixels < self.min_mask_area:
            self.get_logger().warn(
                f"[track_node] Mask too small ({num_pixels} pixels) — ignoring this measurement."
            )
            return

        # Try using centroid of the mask first.
        cx, cy = self.get_centroid_of_mask(mask)
        if cx == -1 or cy == -1 or mask[int(cy), int(cx)] != 255:
            cx, cy = self.get_furthest_point_from_mask_edge(mask)
            if cx == -1 or cy == -1:
                self.get_logger().warn('[track_node] Could not compute a valid point in the mask')
                return

        # Estimate depth from the masked object.
        depth = self.get_median_depth_2(depth_image, mask)

        if depth <= 0:
            return

        now = self.get_clock().now()

        # Convert depth image coordinates to 3D camera space.
        k = cam_info_msg.k
        fx, fy, px, py = k[0], k[4], k[2], k[5]

        z = depth
        x = (int(cx) - px) * z / fx
        y = (int(cy) - py) * z / fy

        measurement_camera = (float(x), float(y), float(z))

        # Transform camera-frame measurement into target_frame.
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time(),
            )
        except Exception as e:
            self.get_logger().warn(
                f"[track_node] Could not transform measurement from {self.camera_frame} "
                f"to {self.target_frame}: {e}",
                throttle_duration_sec=2.0,
            )
            return

        measurement_target = self.transform_point_ros2(
            measurement_camera,
            transform,
        )
        measurement_target = list(measurement_target)

        # For person/ground-plane tracking, fix target-frame height.
        if self.fix_height:
            measurement_target[2] = self.fixed_height

        # Time-gated outlier rejection in target_frame.
        if self.has_measurement:
            predicted = self.ekf.get_state()[:3]

            dxy = np.linalg.norm(
                np.array(predicted[:2], dtype=np.float64)
                - np.array(measurement_target[:2], dtype=np.float64)
            )

            time_since_last = (now - self.last_update_time).nanoseconds * 1e-9

            if dxy > self.max_position_jump:
                if time_since_last < self.relock_window:
                    self.get_logger().warn(
                        f"[track_node] Rejecting large XY jump dxy={dxy:.2f} "
                        f"(only {time_since_last:.2f}s since last valid update)"
                    )
                    return
                else:
                    self.get_logger().info(
                        f"[track_node] Large XY jump dxy={dxy:.2f} but "
                        f"{time_since_last:.2f}s passed — re-locking!"
                    )

        if self.print_measurement_marker:
            self.debug_marker(
                x=float(measurement_target[0]),
                y=float(measurement_target[1]),
                z=float(measurement_target[2]),
                frame_id=self.target_frame,
            )

        # Update EKF in target_frame.
        measurement_np = np.array(
            [
                float(measurement_target[0]),
                float(measurement_target[1]),
                float(measurement_target[2]),
            ],
            dtype=np.float64,
        )

        if not self.has_measurement:
            self.reset_ekf_state(measurement_np)
        else:
            self.ekf.update(
                measurement_np.tolist(),
                dynamic_R=self.measurement_noise_cov,
            )

        self.has_measurement = True
        self.centroid = (float(cx), float(cy))
        self.last_update_time = now


    def debug_marker(self, x: float, y: float, z: float, frame_id: Optional[str] = None):
        # --- Debug marker for measurement ---
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id or self.camera_frame

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



    def transform_point_ros2(self, point: Tuple, transform: TransformStamped) -> Tuple:
        ps = PointStamped()
        ps.header.stamp = transform.header.stamp
        ps.header.frame_id = transform.child_frame_id  # source frame
        ps.point.x = point[0]
        ps.point.y = point[1]
        ps.point.z = point[2]

        # Use the do_transform_point helper
        transformed_ps = do_transform_point(ps, transform)

        return (transformed_ps.point.x, transformed_ps.point.y, transformed_ps.point.z)

    def event_callback(self, msg: String):
        if msg.data == "e_stop":
            self.enable = False
            self.get_logger().info("[track_node] Received e_stop → pausing tracking.")
        elif msg.data == "e_start":
            self.enable = True
            self.has_measurement = False
            self.latest_packet = None
            self.centroid = (0.0, 0.0)
            self.last_update_time = self.get_clock().now()
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
