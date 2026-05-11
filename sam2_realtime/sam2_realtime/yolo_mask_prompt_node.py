#!/usr/bin/env python3

import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class YOLOMaskPromptNode(Node):
    """
    Simple YOLO segmentation prompt node for SAM2.

    Behavior:
      - Waits for e_start on /sam2_mask_prompt/event_in.
      - Stores latest RGB frame from image_topic.
      - Runs YOLO segmentation outside the image callback.
      - Publishes the best person mask to /sam2/init_prompt_mask.
      - Shuts itself down after publishing one prompt.
    """

    def __init__(self):
        super().__init__("yolo_mask_prompt_node")

        # Parameters
        self.declare_parameter("image_topic", "/k4a/rgb/image_raw")
        self.declare_parameter("yolo_model", "yolov8n-seg.pt")
        self.declare_parameter("detect_class", "person")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("min_mask_area", 2000)
        self.declare_parameter("imgsz", 640)

        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("processing_timer_period", 0.001)

        # Load parameters
        self.image_topic = (
            self.get_parameter("image_topic")
            .get_parameter_value()
            .string_value
        )
        yolo_model = (
            self.get_parameter("yolo_model")
            .get_parameter_value()
            .string_value
        )
        self.detect_class = (
            self.get_parameter("detect_class")
            .get_parameter_value()
            .string_value
        )

        self.conf_threshold = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        self.min_mask_area = (
            self.get_parameter("min_mask_area")
            .get_parameter_value()
            .integer_value
        )
        self.imgsz = (
            self.get_parameter("imgsz")
            .get_parameter_value()
            .integer_value
        )

        self.image_reliability = (
            self.get_parameter("image_reliability")
            .get_parameter_value()
            .integer_value
        )
        self.processing_timer_period = (
            self.get_parameter("processing_timer_period")
            .get_parameter_value()
            .double_value
        )

        # Model path
        assets_root = os.environ.get("YOLO_ASSETS_DIR", "")
        if not assets_root:
            raise RuntimeError("YOLO_ASSETS_DIR environment variable not set")

        self.yolo_model_path = os.path.join(assets_root, yolo_model)

        # Init
        self.bridge = CvBridge()
        self.yolo = YOLO(self.yolo_model_path)

        self.class_filter = self.parse_class_spec(self.detect_class)

        if self.class_filter is not None and -1 in self.class_filter:
            self.get_logger().warn(
                f"[yolo_mask_prompt_node] Requested class '{self.detect_class}' "
                "was not found in the YOLO model names. The node will keep waiting "
                "because no valid mask will be detected for that class."
            )

        self.detection_started = False
        self.prompt_sent = False
        self.latest_image_msg = None
        self.processing = False

        # QoS for camera stream
        self.image_qos = QoSProfile(
            reliability=self.image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # ROS interfaces
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            self.image_qos,
        )

        self.publisher = self.create_publisher(
            Image,
            "/sam2/init_prompt_mask",
            10,
        )

        self.event_sub = self.create_subscription(
            String,
            "/sam2_mask_prompt/event_in",
            self.event_cb,
            10,
        )

        self.processing_timer = self.create_timer(
            self.processing_timer_period,
            self.process_latest_image,
        )

        if self.class_filter is None:
            cls_str = "ALL classes"
        elif -1 in self.class_filter:
            cls_str = f"UNKNOWN requested class: {self.detect_class}"
        else:
            cls_str = ", ".join(str(c) for c in self.class_filter)

        self.get_logger().info(
            "[yolo_mask_prompt_node] Ready. "
            "Listening for /sam2_mask_prompt/event_in"
        )
        self.get_logger().info(
            f"[yolo_mask_prompt_node] Model={self.yolo_model_path} | "
            f"Requested class='{self.detect_class}' | "
            f"Resolved classes={cls_str} | "
            f"conf>={self.conf_threshold} | "
            f"min_mask_area={self.min_mask_area} | "
            f"imgsz={self.imgsz}"
        )

    def parse_class_spec(self, spec):
        """
        Convert a class specification into YOLO class IDs.

        Accepted examples:
          - "person"
          - "chair"
          - "person,chair"
          - "0,56"
          - "all"

        Returns:
          - None for all classes
          - list[int] for selected classes

        Unknown class names intentionally resolve to [-1], which produces no valid detections.
        """
        if spec is None:
            return None

        spec = spec.strip()

        if not spec or spec.lower() == "all":
            return None

        yolo_names = self.yolo.names

        if isinstance(yolo_names, dict):
            name_to_id = {
                str(name).lower(): int(idx)
                for idx, name in yolo_names.items()
            }
            max_id = max(yolo_names.keys()) if yolo_names else -1
        else:
            name_to_id = {
                str(name).lower(): idx
                for idx, name in enumerate(yolo_names)
            }
            max_id = len(yolo_names) - 1

        class_ids = []

        for part in spec.split(","):
            part = part.strip()

            if not part:
                continue

            if part.isdigit():
                class_id = int(part)

                if 0 <= class_id <= max_id:
                    class_ids.append(class_id)
                else:
                    class_ids.append(-1)

                continue

            key = " ".join(part.lower().split())

            if key in name_to_id:
                class_ids.append(name_to_id[key])
            else:
                class_ids.append(-1)

        # Remove duplicates, preserving order.
        output = []
        seen = set()

        for class_id in class_ids:
            if class_id not in seen:
                seen.add(class_id)
                output.append(class_id)

        if not output:
            return None

        return output

    def event_cb(self, msg: String):
        if msg.data == "e_start":
            self.detection_started = True
            self.prompt_sent = False
            self.latest_image_msg = None

            self.get_logger().info(
                "[yolo_mask_prompt_node] Received 'e_start'. Starting detection..."
            )

        elif msg.data == "e_stop":
            self.detection_started = False
            self.latest_image_msg = None

            self.get_logger().info(
                "[yolo_mask_prompt_node] Received 'e_stop'. Pausing detection."
            )

        else:
            self.get_logger().warn(
                f"[yolo_mask_prompt_node] Unknown event: '{msg.data}'"
            )

    def image_callback(self, msg: Image):
        """
        Store only the latest image.

        YOLO segmentation is done in process_latest_image(), not in this callback.
        """
        if not self.detection_started or self.prompt_sent:
            return

        self.latest_image_msg = msg

    def process_latest_image(self):
        """
        Process latest image and publish one SAM2 mask prompt.

        After publishing the prompt, the node shuts down because this node is
        intended to be a one-shot SAM2 prompt provider.
        """
        if self.processing:
            return

        if not self.detection_started or self.prompt_sent:
            self.latest_image_msg = None
            return

        if self.latest_image_msg is None:
            return

        msg = self.latest_image_msg
        self.latest_image_msg = None

        self.processing = True

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            results = self.yolo.predict(
                frame,
                classes=self.class_filter,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )

            best_mask = None
            max_area = 0

            try:
                r0 = results[0]
                if r0.boxes is not None:
                    classes_list = [int(c) for c in r0.boxes.cls.tolist()]
                    confs_list = [float(c) for c in r0.boxes.conf.tolist()]
                    self.get_logger().info(
                        f"[yolo_mask_prompt_node] Detections: {len(classes_list)} | "
                        f"classes={classes_list} | "
                        f"confs={[round(c, 3) for c in confs_list]}"
                    )
            except Exception:
                pass

            for result in results:
                if result.masks is None or result.boxes is None:
                    continue

                for i, box in enumerate(result.boxes):
                    conf = float(box.conf[0])

                    if conf < self.conf_threshold:
                        continue

                    mask = (
                        result.masks.data[i]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                        * 255
                    )

                    # Count foreground pixels. np.sum(mask) would sum 255 values,
                    # so countNonZero is clearer for area in pixels.
                    area = int(np.count_nonzero(mask))

                    if area < self.min_mask_area:
                        continue

                    if area > max_area:
                        best_mask = mask
                        max_area = area

            if best_mask is None:
                self.get_logger().info(
                    f"[yolo_mask_prompt_node] No valid mask found for "
                    f"'{self.detect_class}'. Waiting for next frame..."
                )
                return

            ros_mask = self.bridge.cv2_to_imgmsg(
                best_mask,
                encoding="mono8",
            )
            ros_mask.header = msg.header

            self.publisher.publish(ros_mask)

            self.prompt_sent = True
            self.detection_started = False

            self.get_logger().info(
                f"[yolo_mask_prompt_node] ✅ Published mask with area={max_area} pixels"
            )
            self.get_logger().info(
                "[yolo_mask_prompt_node] Prompt sent. Shutting down..."
            )

            rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(
                f"[yolo_mask_prompt_node] Processing failed: {e}"
            )

        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)

    node = YOLOMaskPromptNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()