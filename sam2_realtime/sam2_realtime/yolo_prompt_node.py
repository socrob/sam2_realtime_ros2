#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sam2_realtime_msgs.msg import PromptBbox
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
import os


class YOLOBboxPromptNode(Node):
    def __init__(self):
        super().__init__('yolo_bbox_prompt_node')

        self.declare_parameter('image_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.9)
        self.declare_parameter('min_box_area', 2000)
        self.declare_parameter('max_aspect_ratio', 3.0)

        # Load parameters
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.min_box_area = self.get_parameter('min_box_area').get_parameter_value().integer_value
        self.max_aspect_ratio = self.get_parameter('max_aspect_ratio').get_parameter_value().double_value

        assets_root = os.environ.get("YOLO_ASSETS_DIR", "")
        if not assets_root:
            raise RuntimeError("YOLO_ASSETS_DIR environment variable not set")
        self.yolo_model_path = os.path.join(assets_root, yolo_model)

        self.bridge = CvBridge()
        self.yolo = YOLO(self.yolo_model_path)

        self.detection_started = False
        self.prompt_sent = False

        # ROS interfaces
        self.subscription = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(PromptBbox, '/sam2/init_prompt', 10)
        self.create_subscription(String, '/sam2_bbox_prompt/event_in', self.event_cb, 10)

        self.get_logger().info(f'[yolo_bbox_prompt_node] Ready. Listening for event on /sam2_bbox_prompt/event_in')

    def event_cb(self, msg: String):
        if msg.data == 'e_start':
            self.detection_started = True
            self.get_logger().info("[yolo_bbox_prompt_node] Received 'e_start'. Starting detection...")

    def image_callback(self, msg: Image):
        if not self.detection_started or self.prompt_sent:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.yolo.predict(frame, classes=[0], conf=self.conf_threshold)

        best_box = None
        max_area = 0

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != 0:
                    continue

                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                area = w * h

                if area < self.min_box_area:
                    continue

                aspect_ratio = max(h / w, w / h)
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
            self.get_logger().info(f'[yolo_bbox_prompt_node] ✅ Published bbox: {best_box}')
            self.get_logger().info('[yolo_bbox_prompt_node] Shutting down...')
            rclpy.shutdown()
        else:
            self.get_logger().info('[yolo_bbox_prompt_node] No valid bbox found. Waiting for next frame...')


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
