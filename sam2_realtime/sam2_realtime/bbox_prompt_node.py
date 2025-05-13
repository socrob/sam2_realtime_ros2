#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sam2_realtime_msgs.msg import PromptBbox
from cv_bridge import CvBridge
import numpy as np
import cv2

class BBoxPromptNode(Node):

    def __init__(self):
        super().__init__('bbox_prompt_node')

        # Declare and read image topic
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        # ROS interfaces
        self.prompt_pub = self.create_publisher(PromptBbox, '/sam2/init_prompt', 10)
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        self.bridge = CvBridge()

        # Interaction state
        self.frame = None
        self.drawing = False
        self.ix, self.iy, self.fx, self.fy = -1, -1, -1, -1
        self.bbox = None
        self.enter_pressed = False

        cv2.namedWindow("Prompt Selector")
        cv2.setMouseCallback("Prompt Selector", self.draw_rectangle)

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.fx, self.fy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fx, self.fy = x, y
            self.bbox = (self.ix, self.iy, self.fx, self.fy)
            self.enter_pressed = True

    def image_callback(self, msg: Image):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        display_frame = self.frame.copy()

        if not self.enter_pressed:
            if self.drawing:
                cv2.rectangle(display_frame, (self.ix, self.iy), (self.fx, self.fy), (255, 0, 0), 2)
            cv2.putText(display_frame, "Draw a box to send prompt", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            x0, y0 = int(min(self.ix, self.fx)), int(min(self.iy, self.fy))
            x1, y1 = int(max(self.ix, self.fx)), int(max(self.iy, self.fy))
            cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
            prompt = PromptBbox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)
            self.prompt_pub.publish(prompt)
            self.get_logger().info(f"Published PromptBbox: {prompt}")
            self.enter_pressed = False

        cv2.imshow("Prompt Selector", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = BBoxPromptNode()
    rclpy.spin(node)
    rclpy.shutdown()
