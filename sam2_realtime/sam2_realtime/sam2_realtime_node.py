#!/usr/bin/env python3

import traceback
import rclpy
from rclpy.clock import Clock
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from sam2_realtime_msgs.msg import PromptBbox
from cv_bridge import CvBridge
from sam2_realtime_msgs.msg import TrackedObject
import torch
import numpy as np
import cv2

from sam2.build_sam import build_sam2_camera_predictor


class SAM2Node(LifecycleNode):

    def __init__(self):
        super().__init__('sam2_node')

        # Declare parameters
        self.declare_parameter('image_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('model_cfg', 'configs/sam2.1/sam2.1_hiera_s.yaml')
        self.declare_parameter('checkpoint', 'checkpoints/sam2.1_hiera_small.pt')
        self.declare_parameter('image_reliability', QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter('live_visualization', False)
        self.declare_parameter('processing_timer_period', 0.001)

        # Runtime flags
        self.predictor = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            self.get_logger().info('[sam2_node] Configuring...')

            # Load parameters
            self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
            self.model_cfg = self.get_parameter('model_cfg').get_parameter_value().string_value
            self.checkpoint = self.get_parameter('checkpoint').get_parameter_value().string_value
            self.reliability = self.get_parameter('image_reliability').get_parameter_value().integer_value
            self.live_visualization = self.get_parameter('live_visualization').get_parameter_value().bool_value
            self.processing_timer_period = self.get_parameter('processing_timer_period').get_parameter_value().double_value
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


            self.cv_bridge = CvBridge()

            # Lifecycle Publisher
            self._pub = self.create_lifecycle_publisher(TrackedObject, "mask", 10)
            self._img_pub = self.create_lifecycle_publisher(Image, "img_mask", 10)
            
            # Runtime state
            self.latest_image_msg = None
            self.processing = False

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
            
            # Load SAM2 predictor
            try:
                self.predictor = build_sam2_camera_predictor(self.model_cfg, self.checkpoint)
            except FileNotFoundError:
                self.get_logger().error(f"Model file '{self.model_cfg}' does not exists")
                return TransitionCallbackReturn.ERROR

            self.initialized = False
            self.bbox = None
            self.mask_prompt = None
            self.latest_image_msg = None
            self.processing = False

            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_cb, self.qos)
            self.prompt_sub = self.create_subscription(PromptBbox, 'init_prompt', self.prompt_cb, 10)
            self.prompt_mask_sub = self.create_subscription(Image, 'init_prompt_mask', self.prompt_mask_cb, 10)
            
            self._timer = self.create_timer(self.processing_timer_period, self.process_latest_image)

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

            if hasattr(self, "_timer") and self._timer is not None:
                self.destroy_timer(self._timer)
                self._timer = None

            if hasattr(self, "image_sub") and self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None

            if hasattr(self, "prompt_sub") and self.prompt_sub is not None:
                self.destroy_subscription(self.prompt_sub)
                self.prompt_sub = None

            if hasattr(self, "prompt_mask_sub") and self.prompt_mask_sub is not None:
                self.destroy_subscription(self.prompt_mask_sub)
                self.prompt_mask_sub = None

            if hasattr(self, "predictor") and self.predictor is not None:
                del self.predictor
                self.predictor = None

            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

            self.initialized = False
            self.bbox = None
            self.mask_prompt = None
            self.latest_image_msg = None
            self.processing = False

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
            
            self.initialized = False
            self.bbox = None
            self.mask_prompt = None
            
            # Destroy Publisher
            self.destroy_publisher(self._pub)
            self.destroy_publisher(self._img_pub)
            
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
            super().on_shutdown(state)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"[sam2_node] Exception during shutdown: {e}")
            self.get_logger().error(traceback.format_exc())
            return TransitionCallbackReturn.FAILURE
        
    
    def destroy_prompt_subscribers(self):
        if hasattr(self, "prompt_sub") and self.prompt_sub is not None:
            self.destroy_subscription(self.prompt_sub)
            self.prompt_sub = None

        if hasattr(self, "prompt_mask_sub") and self.prompt_mask_sub is not None:
            self.destroy_subscription(self.prompt_mask_sub)
            self.prompt_mask_sub = None

        self.get_logger().info("[sam2_node] Destroyed prompt subscribers!")


    def prompt_cb(self, msg: PromptBbox):
        self.bbox = (msg.x_min, msg.y_min, msg.x_max, msg.y_max)
        self.get_logger().info(f"[sam2_node] Received init prompt: {self.bbox}")


    def prompt_mask_cb(self, msg: Image):
        self.mask_prompt = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.get_logger().info("[sam2_node] Received mask prompt")


    def image_cb(self, msg: Image):
        """
        Store only the latest image.

        Heavy SAM2 initialization/tracking is done in process_latest_image().
        """
        self.latest_image_msg = msg
    
    def process_latest_image(self):
        """
        Process the latest image, if available.

        This keeps the image callback lightweight and avoids processing stale frames
        if SAM2 tracking is slower than the camera stream.
        """
        if self.processing:
            return

        if self.latest_image_msg is None:
            return

        # Do not even convert the image until we have a prompt.
        if self.bbox is None and self.mask_prompt is None:
            return

        msg = self.latest_image_msg
        self.latest_image_msg = None

        self.processing = True

        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width = frame.shape[:2]

            if not self.initialized:
                if self.mask_prompt is not None:
                    mask_normalized = self.mask_prompt.astype(np.float32) / 255.0

                    self.predictor.load_first_frame(frame)
                    _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_mask(
                        frame_idx=0,
                        obj_id=1,
                        mask=mask_normalized
                    )

                    self.get_logger().info("[sam2_node] Initialized with mask prompt")

                else:
                    bbox_np = np.array(
                        [
                            [self.bbox[0], self.bbox[1]],
                            [self.bbox[2], self.bbox[3]]
                        ],
                        dtype=np.float32
                    )

                    self.predictor.load_first_frame(frame)
                    _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=1,
                        bbox=bbox_np
                    )

                    self.get_logger().info("[sam2_node] Initialized with bbox prompt")

                self.initialized = True

            else:
                self.out_obj_ids, self.out_mask_logits = self.predictor.track(frame)

                all_mask = np.zeros((height, width), dtype=np.uint8)

                for i in range(len(self.out_obj_ids)):
                    out_mask = (self.out_mask_logits[i] > 0.0)
                    out_mask = out_mask.squeeze().detach().cpu().numpy().astype(np.uint8) * 255

                    # If multiple objects are tracked later, merge masks instead of overwriting.
                    all_mask = np.maximum(all_mask, out_mask)

                # Publish msg
                sam2_msg = TrackedObject()
                sam2_msg.header.stamp = msg.header.stamp
                sam2_msg.header.frame_id = msg.header.frame_id
                sam2_msg.id = 1

                sam2_msg.mask = self.cv_bridge.cv2_to_imgmsg(all_mask, encoding='mono8')
                sam2_msg.mask.header.stamp = msg.header.stamp
                sam2_msg.mask.header.frame_id = msg.header.frame_id

                self._pub.publish(sam2_msg)

                if self.live_visualization:
                    mask_rgb = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
                    vis_frame = cv2.addWeighted(frame, 1.0, mask_rgb, 0.5, 0.0)

                    debug_msg = self.cv_bridge.cv2_to_imgmsg(vis_frame, encoding="bgr8")
                    debug_msg.header.stamp = msg.header.stamp
                    debug_msg.header.frame_id = msg.header.frame_id
                    self._img_pub.publish(debug_msg)

                return

            # Optional visualization after initialization frame.
            if self.live_visualization:
                debug_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                debug_msg.header.stamp = msg.header.stamp
                debug_msg.header.frame_id = msg.header.frame_id
                self._img_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"[sam2_node] Processing failed: {e}")
            self.get_logger().error(traceback.format_exc())

        finally:
            self.processing = False
        
        
        
    # def image_cb(self, msg: Image):
    #     frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     height, width = frame.shape[:2]

    #     if self.bbox is None and self.mask_prompt is None:
    #         return

    #     if not self.initialized:
    #         if self.mask_prompt is not None:
    #             mask_normalized = self.mask_prompt.astype(np.float32) / 255.0
    #             self.predictor.load_first_frame(frame)
    #             _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_mask(
    #                 frame_idx=0, obj_id=1, mask=mask_normalized
    #             )
    #             self.get_logger().info("[sam2_node] Initialized with mask prompt")
    #         else:
    #             bbox_np = np.array([[self.bbox[0], self.bbox[1]], [self.bbox[2], self.bbox[3]]], dtype=np.float32)
    #             self.predictor.load_first_frame(frame)
    #             _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_prompt(
    #                 frame_idx=0, obj_id=1, bbox=bbox_np
    #             )
    #             self.get_logger().info("[sam2_node] Initialized with bbox prompt")
                
    #         self.initialized = True
    #     else:
    #         self.out_obj_ids, self.out_mask_logits = self.predictor.track(frame)
    #         all_mask = np.zeros((height, width, 1), dtype=np.uint8)
    #         for i in range(len(self.out_obj_ids)):
    #             out_mask = (self.out_mask_logits[i] > 0.0).permute(1, 2, 0).byte().cuda()
    #             all_mask = out_mask.cpu().numpy() * 255


    #         # Publish msg
    #         sam2_msg = TrackedObject()
    #         now = Clock().now().to_msg()

    #         # sam2_msg.header.stamp = now
    #         sam2_msg.header.stamp = msg.header.stamp
    #         sam2_msg.header.frame_id = msg.header.frame_id
    #         sam2_msg.id = 1

    #         sam2_msg.mask = self.cv_bridge.cv2_to_imgmsg(all_mask, encoding='mono8')
    #         # sam2_msg.mask.header.stamp = now
    #         sam2_msg.header.stamp = msg.header.stamp
    #         sam2_msg.mask.header.frame_id = msg.header.frame_id

    #         self._pub.publish(sam2_msg)

    #         all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
    #         frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

    #     # Display result
    #     if self.live_visualization:
    #         debug_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8")
    #         now = Clock().now().to_msg()
    #         debug_msg.header.stamp = now
    #         debug_msg.header.frame_id = msg.header.frame_id
    #         self._img_pub.publish(debug_msg)
    #         # Live Visualization
    #         # cv2.imshow("SAM2 Tracking", frame)
    #         # cv2.waitKey(1)


def main():
    rclpy.init()
    node = SAM2Node()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
