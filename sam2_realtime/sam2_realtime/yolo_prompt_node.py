#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from sam2_realtime_msgs.msg import PromptBbox
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
import os
from typing import List, Dict

# COCO 80-class names (YOLOv8 order, 0-based indexing)
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(COCO_NAMES)}

def parse_class_spec(spec: str) -> List[int]:
    if spec is None:
        return [0]  # default to person
    spec = spec.strip()
    if spec.lower() == "all":
        return []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    ids: List[int] = []
    for p in parts:
        if p.isdigit():
            cid = int(p)
            if 0 <= cid < len(COCO_NAMES):
                ids.append(cid)
            else:
                raise ValueError(f"Class id out of range: {cid}")
        else:
            key = p.lower()
            synonyms = {
                "tv": "tv","tvmonitor": "tv","cellphone": "cell phone","mobile": "cell phone","mobile phone": "cell phone",
                "aeroplane": "airplane","sofa": "couch","diningtable": "dining table","pottedplant": "potted plant","hairdryer": "hair drier",
            }
            if key in synonyms:
                key = synonyms[key]
            key = " ".join(key.split())
            if key in NAME_TO_ID:
                ids.append(NAME_TO_ID[key])
            else:
                raise ValueError(f"Unknown class name: '{p}'")
    seen = set()
    out = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


class YOLOBboxPromptNode(Node):
    def __init__(self):
        super().__init__('yolo_bbox_prompt_node')

        self.declare_parameter('image_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('detect_class', 'person')
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter('min_box_area', 800)
        self.declare_parameter('max_aspect_ratio', 3.0)
        self.declare_parameter('imgsz', 640)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        detect_class = self.get_parameter('detect_class').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.min_box_area = self.get_parameter('min_box_area').get_parameter_value().integer_value
        self.max_aspect_ratio = self.get_parameter('max_aspect_ratio').get_parameter_value().double_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value

        try:
            class_ids = parse_class_spec(detect_class)
            self.class_filter = None if len(class_ids) == 0 else class_ids
        except ValueError as e:
            raise RuntimeError(f"[yolo_bbox_prompt_node] {e} — valid names include: {', '.join(COCO_NAMES[:10])} ...")

        assets_root = os.environ.get("YOLO_ASSETS_DIR", "")
        if not assets_root:
            raise RuntimeError("YOLO_ASSETS_DIR environment variable not set")
        self.yolo_model_path = os.path.join(assets_root, yolo_model)

        self.bridge = CvBridge()
        self.yolo = YOLO(self.yolo_model_path)

        self.detection_started = False
        self.prompt_sent = False

        self.subscription = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(PromptBbox, '/sam2/init_prompt', 10)
        self.create_subscription(String, '/sam2_bbox_prompt/event_in', self.event_cb, 10)

        if self.class_filter is None:
            cls_str = "ALL classes"
        else:
            cls_str = ", ".join(f"{cid}:{COCO_NAMES[cid]}" for cid in self.class_filter)
        self.get_logger().info(f"[yolo_bbox_prompt_node] Ready. Listening for /sam2_bbox_prompt/event_in")
        self.get_logger().info(f"[yolo_bbox_prompt_node] Model={self.yolo_model_path} | Classes={cls_str} | conf>={self.conf_threshold} | imgsz={self.imgsz}")

    def event_cb(self, msg: String):
        if msg.data == 'e_start':
            self.detection_started = True
            self.get_logger().info("[yolo_bbox_prompt_node] Received 'e_start'. Starting detection...")

    def image_callback(self, msg: Image):
        if not self.detection_started or self.prompt_sent:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.yolo.predict(
            frame,
            classes=self.class_filter,
            conf=self.conf_threshold,
            imgsz=self.imgsz
        )

        best_box = None
        max_area = 0

        try:
            r0 = results[0]
            classes_list = [int(c) for c in r0.boxes.cls.tolist()]
            confs_list = [float(c) for c in r0.boxes.conf.tolist()]
            self.get_logger().info(f"Detections: {len(classes_list)} | classes={classes_list} | confs={[round(c,3) for c in confs_list]}")
        except Exception:
            pass

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                if area < self.min_box_area:
                    continue

                aspect_ratio = max(h / w, w / h) if w > 0 and h > 0 else 999.0
                if aspect_ratio > self.max_aspect_ratio:
                    continue

                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

        if best_box:
            prompt = PromptBbox()
            prompt.x_min, prompt.y_min, prompt.x_max, prompt.y_max = best_box
            self.publisher.publish(prompt)
            self.prompt_sent = True
            self.get_logger().info(f"[yolo_bbox_prompt_node] ✅ Published bbox: {best_box}")
            self.get_logger().info("[yolo_bbox_prompt_node] Shutting down...")
            rclpy.shutdown()
        else:
            self.get_logger().info("[yolo_bbox_prompt_node] No valid bbox found. Waiting for next frame...")


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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
