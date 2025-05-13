#!/usr/bin/env python3

import traceback
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from sam2_realtime_msgs.msg import PromptBbox
from cv_bridge import CvBridge

import torch
import numpy as np
import cv2, os

from sam2.build_sam import build_sam2_camera_predictor


class SAM2Node(LifecycleNode):

    def __init__(self):
        super().__init__('sam2_node')

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('image_reliability', QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter('model_cfg', 'configs/sam2.1/sam2.1_hiera_s.yaml')
        self.declare_parameter('checkpoint', 'checkpoints/sam2.1_hiera_small.pt')

        # Runtime flags
        self.initialized = False
        self.predictor = None
        self.bbox = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[sam2_node] Configuring...')

            # Load parameters
            self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
            self.model_cfg = self.get_parameter('model_cfg').get_parameter_value().string_value
            self.checkpoint = self.get_parameter('checkpoint').get_parameter_value().string_value
            self.reliability = self.get_parameter('image_reliability').get_parameter_value().integer_value

            self.get_logger().info(f'[sam2_node] Using config: {self.model_cfg}')
            self.get_logger().info(f'[sam2_node] Using checkpoint: {self.checkpoint}')

            # Setup QoS
            self.qos = QoSProfile(
                reliability=self.reliability,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1
            )

            # Setup Torch
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Load SAM2 predictor
            self.predictor = build_sam2_camera_predictor(self.model_cfg, self.checkpoint)

            self.cv_bridge = CvBridge()
            
            # Lifecycle Publisher
            #TODO

            super().on_configure(state)
            self.get_logger().info('[sam2_node] SAM2 model loaded')

            return TransitionCallbackReturn.SUCCESS
        
        except Exception as e:
            self.get_logger().error(f"[sam2_node] Exception during configuration: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[sam2_node] Activating...')
            
            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_cb, self.qos)
            self.prompt_sub = self.create_subscription(PromptBbox, '/sam2/init_prompt', self.prompt_cb, 10)

            super().on_activate(state)
            self.get_logger().info('[sam2_node] Subscriptions activated')
            return TransitionCallbackReturn.SUCCESS
        
        except Exception as e:
            self.get_logger().error(f"[sam2_node] Exception during activation: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[sam2_node] Deactivating...')
            self.destroy_subscription(self.image_sub)
            self.destroy_subscription(self.prompt_sub)
            super().on_deactivate(state)
            self.get_logger().info('[sam2_node] Deactivated')
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[sam2_node] Exception during deactivation: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[sam2_node] Cleaning up...')
            self.predictor = None
            self.initialized = False
            # Destroy Publisher
            del self.qos
            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[sam2_node] Exception during cleanup: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[sam2_node] Shutting down...')
            super().on_cleanup(state)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[sam2_node] Exception during shutdown: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE

    def prompt_cb(self, msg: PromptBbox):
        self.bbox = (msg.x_min, msg.y_min, msg.x_max, msg.y_max)
        self.get_logger().info(f"[sam2_node] Received init prompt: {self.bbox}")
        self.initialized = False  # Force reinit in next frame

    def image_cb(self, msg: Image):
        frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]
        frame = cv2.flip(frame, 1)

        if self.bbox is None:
            return  # Waiting for prompt

        if not self.initialized:
            bbox_np = np.array([[self.bbox[0], self.bbox[1]], [self.bbox[2], self.bbox[3]]], dtype=np.float32)
            self.predictor.load_first_frame(frame)
            _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0, obj_id=1, bbox=bbox_np)
            self.initialized = True
            self.get_logger().info("[sam2_node] SAM2 tracking initialized")

        else:
            self.out_obj_ids, self.out_mask_logits = self.predictor.track(frame)
            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            for i in range(len(self.out_obj_ids)):
                out_mask = (self.out_mask_logits[i] > 0.0).permute(1, 2, 0).byte().cuda()
                all_mask = out_mask.cpu().numpy() * 255

            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

        # Display result (debug only)
        cv2.imshow("SAM2 Tracking", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = SAM2Node()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
