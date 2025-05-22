#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sam2_realtime_msgs.msg import PromptBbox
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
import os

class YOLOPromptNode(Node):
    def __init__(self):
        super().__init__('yolo_prompt_node')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        # self.declare_parameter('image_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        assets_root = os.environ.get("YOLO_ASSETS_DIR", "")
        if not assets_root:
            raise RuntimeError("YOLO_ASSETS_DIR environment variable not set")
        
        self.yolo_model_path = os.path.join(assets_root, yolo_model)

        self.bridge = CvBridge()
        self.yolo = YOLO(self.yolo_model_path)
        self.conf_threshold = conf_threshold

        self.subscription = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(PromptBbox, '/sam2/init_prompt', 10)

        self.get_logger().info(f'[yolo_prompt_node] Listening on {image_topic}, YOLO model: {yolo_model}')

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.yolo.predict(frame, classes=[0], conf=self.conf_threshold)  # Class 0 = person

        best_box = None
        max_area = 0

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

        if best_box:
            prompt = PromptBbox()
            prompt.x_min, prompt.y_min, prompt.x_max, prompt.y_max = best_box
            self.publisher.publish(prompt)
            self.get_logger().info(f'[yolo_prompt_node] Published bbox: {best_box}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOPromptNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
