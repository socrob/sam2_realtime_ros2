#!/usr/bin/env python3

import os
from typing import Optional, List

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

from sam2_realtime_msgs.msg import PromptBbox


def parse_class_spec(spec: str, yolo_names) -> Optional[List[int]]:
    """
    Convert a class specification into YOLO class IDs.

    Accepted examples:
      - "person"
      - "person,chair"
      - "0,56"
      - "all"

    Returns:
      - None for all classes
      - list[int] for selected classes

    Unknown class names are allowed at startup, but they will produce no detections.
    """
    if spec is None:
        return None

    spec = spec.strip()

    if not spec or spec.lower() == "all":
        return None

    # Ultralytics usually exposes names as dict[int, str].
    if isinstance(yolo_names, dict):
        name_to_id = {str(name).lower(): int(idx) for idx, name in yolo_names.items()}
        max_id = max(yolo_names.keys()) if yolo_names else -1
    else:
        name_to_id = {str(name).lower(): idx for idx, name in enumerate(yolo_names)}
        max_id = len(yolo_names) - 1

    class_ids: List[int] = []

    for part in spec.split(","):
        part = part.strip()

        if not part:
            continue

        if part.isdigit():
            class_id = int(part)
            if 0 <= class_id <= max_id:
                class_ids.append(class_id)
            else:
                # Invalid ID: use an impossible class so it simply detects nothing.
                class_ids.append(-1)
            continue

        key = " ".join(part.lower().split())

        if key in name_to_id:
            class_ids.append(name_to_id[key])
        else:
            # Unknown class name: use an impossible class so it simply detects nothing.
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


class YOLOBboxPromptNode(Node):
    """
    Simple YOLO-based prompt node for SAM2.

    Behavior:
      - Waits for e_start on /sam2_bbox_prompt/event_in.
      - Stores latest RGB frame from image_topic.
      - Runs YOLO outside the image callback.
      - Publishes the best bbox to /sam2/init_prompt.
      - Shuts itself down after publishing one prompt.
    """

    def __init__(self):
        super().__init__("yolo_bbox_prompt_node")

        # Parameters
        self.declare_parameter("image_topic", "/k4a/rgb/image_raw")
        self.declare_parameter("yolo_model", "yolov8n.pt")
        self.declare_parameter("detect_class", "person")
        self.declare_parameter("confidence_threshold", 0.4)
        self.declare_parameter("min_box_area", 800)
        self.declare_parameter("max_aspect_ratio", 3.0)
        self.declare_parameter("imgsz", 640)

        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("processing_timer_period", 0.001)

        # Read parameters
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
        self.min_box_area = (
            self.get_parameter("min_box_area")
            .get_parameter_value()
            .integer_value
        )
        self.max_aspect_ratio = (
            self.get_parameter("max_aspect_ratio")
            .get_parameter_value()
            .double_value
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

        # Runtime state
        self.bridge = CvBridge()
        self.yolo = YOLO(self.yolo_model_path)

        self.class_filter = parse_class_spec(self.detect_class, self.yolo.names)

        if self.class_filter is not None and -1 in self.class_filter:
            self.get_logger().warn(
                f"[yolo_bbox_prompt_node] Requested class '{self.detect_class}' "
                "was not found in the YOLO model names. The node will keep waiting "
                "because no valid bbox will be detected for that class."
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
            PromptBbox,
            "/sam2/init_prompt",
            10,
        )

        self.event_sub = self.create_subscription(
            String,
            "/sam2_bbox_prompt/event_in",
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
            "[yolo_bbox_prompt_node] Ready. "
            "Listening for /sam2_bbox_prompt/event_in"
        )
        self.get_logger().info(
            f"[yolo_bbox_prompt_node] Model={self.yolo_model_path} | "
            f"Requested class='{self.detect_class}' | "
            f"Resolved classes={cls_str} | "
            f"conf>={self.conf_threshold} | "
            f"imgsz={self.imgsz}"
        )

    def event_cb(self, msg: String):
        if msg.data == "e_start":
            self.detection_started = True
            self.prompt_sent = False
            self.latest_image_msg = None
            self.get_logger().info(
                "[yolo_bbox_prompt_node] Received 'e_start'. Starting detection..."
            )

        elif msg.data == "e_stop":
            self.detection_started = False
            self.latest_image_msg = None
            self.get_logger().info(
                "[yolo_bbox_prompt_node] Received 'e_stop'. Pausing detection."
            )

        else:
            self.get_logger().warn(
                f"[yolo_bbox_prompt_node] Unknown event: '{msg.data}'"
            )

    def image_callback(self, msg: Image):
        """
        Store only the latest image.

        YOLO inference is done in process_latest_image(), not in this callback.
        """
        if not self.detection_started or self.prompt_sent:
            return

        self.latest_image_msg = msg

    def process_latest_image(self):
        """
        Process latest image and publish one SAM2 bbox prompt.

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

            # If class_filter contains -1, this intentionally produces no results.
            results = self.yolo.predict(
                frame,
                classes=self.class_filter,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )

            best_box = None
            max_area = 0

            try:
                r0 = results[0]
                classes_list = [int(c) for c in r0.boxes.cls.tolist()]
                confs_list = [float(c) for c in r0.boxes.conf.tolist()]
                self.get_logger().info(
                    f"[yolo_bbox_prompt_node] Detections: {len(classes_list)} | "
                    f"classes={classes_list} | "
                    f"confs={[round(c, 3) for c in confs_list]}"
                )
            except Exception:
                pass

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])

                    if conf < self.conf_threshold:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    w = x2 - x1
                    h = y2 - y1

                    if w <= 0 or h <= 0:
                        continue

                    area = w * h

                    if area < self.min_box_area:
                        continue

                    aspect_ratio = max(h / w, w / h)

                    if aspect_ratio > self.max_aspect_ratio:
                        continue

                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2, y2)

            if best_box is None:
                self.get_logger().info(
                    f"[yolo_bbox_prompt_node] No valid bbox found for "
                    f"'{self.detect_class}'. Waiting for next frame..."
                )
                return

            prompt = PromptBbox()
            prompt.x_min = int(best_box[0])
            prompt.y_min = int(best_box[1])
            prompt.x_max = int(best_box[2])
            prompt.y_max = int(best_box[3])

            self.publisher.publish(prompt)

            self.prompt_sent = True
            self.detection_started = False

            self.get_logger().info(
                f"[yolo_bbox_prompt_node] ✅ Published bbox: {best_box}"
            )
            self.get_logger().info(
                "[yolo_bbox_prompt_node] Prompt sent. Shutting down..."
            )

            rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(
                f"[yolo_bbox_prompt_node] Processing failed: {e}"
            )

        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)

    node = YOLOBboxPromptNode()
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