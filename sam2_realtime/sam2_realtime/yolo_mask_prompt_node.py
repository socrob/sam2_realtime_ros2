#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import os
from ultralytics import YOLO

class YOLOMaskPromptNode(Node):
    def __init__(self):
        super().__init__('yolo_mask_prompt_node')

        # self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('image_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('yolo_model', 'yolov8n-seg.pt')
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
        self.publisher = self.create_publisher(Image, '/sam2/init_prompt_mask', 10)

        self.get_logger().info(f'[yolo_mask_prompt_node] Listening on {image_topic}, YOLO model: {yolo_model}')

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.yolo.predict(frame, classes=[0], conf=self.conf_threshold)

        best_mask = None
        max_area = 0

        for result in results:
            if result.masks is None:
                continue
            for i, box in enumerate(result.boxes):
                if int(box.cls[0]) != 0:
                    continue
                mask = result.masks.data[i].cpu().numpy().astype(np.uint8) * 255
                area = np.sum(mask)
                if area > max_area:
                    best_mask = mask
                    max_area = area

        if best_mask is not None:
            ros_mask = self.bridge.cv2_to_imgmsg(best_mask, encoding="mono8")
            ros_mask.header = msg.header
            self.publisher.publish(ros_mask)
            self.get_logger().info('[yolo_mask_prompt_node] Published mask for closest person')


def main(args=None):
    rclpy.init(args=args)
    node = YOLOMaskPromptNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
