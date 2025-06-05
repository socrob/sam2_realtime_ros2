#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
import os
import sys

from std_msgs.msg import String


class YOLOMaskPromptNode(Node):
    def __init__(self):
        super().__init__('yolo_mask_prompt_node')

        # self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('image_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('yolo_model', 'yolov8n-seg.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('min_mask_area', 2000)

        # Load parameters
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.min_mask_area = self.get_parameter('min_mask_area').get_parameter_value().integer_value

        # Model path
        assets_root = os.environ.get("YOLO_ASSETS_DIR", "")
        if not assets_root:
            raise RuntimeError("YOLO_ASSETS_DIR environment variable not set")
        self.yolo_model_path = os.path.join(assets_root, yolo_model)

        # Init
        self.bridge = CvBridge()
        self.yolo = YOLO(self.yolo_model_path)
        self.detection_started = False
        self.prompt_sent = False

        # ROS interfaces
        self.subscription = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(Image, '/sam2/init_prompt_mask', 10)
        self.create_subscription(String, '/sam2_mask_prompt/event_in', self.event_cb, 10)

        self.get_logger().info(f'[yolo_mask_prompt_node] Ready. Listening for event on /sam2_mask_prompt/event_in')

    def event_cb(self, msg: String):
        if msg.data == 'e_start':
            self.detection_started = True
            self.get_logger().info("[yolo_mask_prompt_node] Received 'e_start'. Starting detection...")

    def image_callback(self, msg: Image):
        if not self.detection_started or self.prompt_sent:
            return

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
                if float(box.conf[0]) < self.conf_threshold:
                    continue
                mask = result.masks.data[i].cpu().numpy().astype(np.uint8) * 255
                area = np.sum(mask)
                if area > self.min_mask_area and area > max_area:
                    best_mask = mask
                    max_area = area

        if best_mask is not None:
            ros_mask = self.bridge.cv2_to_imgmsg(best_mask, encoding="mono8")
            ros_mask.header = msg.header
            self.publisher.publish(ros_mask)
            self.prompt_sent = True
            self.get_logger().info('[yolo_mask_prompt_node] ✅ Published mask for closest person')
            self.get_logger().info('[yolo_mask_prompt_node] Shutting down...')
            rclpy.shutdown()
        else:
            self.get_logger().info('[yolo_mask_prompt_node] No valid mask found. Waiting for next frame...')


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
        rclpy.shutdown()
