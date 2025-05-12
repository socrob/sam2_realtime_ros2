import torch
import numpy as np
import cv2
import pyrealsense2 as rs
from sam2.build_sam import build_sam2_camera_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Set up the SAM2 predictor
sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# Set up the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Set resolution and frame rate
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

pipeline.start(config)

# Initialize the variables
drawing = False
ix, iy, fx, fy = -1, -1, -1, -1
bbox = None
enter_pressed = False

# Mouse callback function for selecting bounding box
def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, bbox, enter_pressed
    if event == cv2.EVENT_LBUTTONDOWN:  # Mouse left button down
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse move
        if drawing:
            fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:  # Mouse left button up
        drawing = False
        fx, fy = x, y
        bbox = (ix, iy, fx, fy)  # Save the bounding box
        enter_pressed = True  # Proceed to next step once drawing is done

# Set up the window and mouse callback
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", draw_rectangle)

if_init = False

while True:
    # Capture frames from the RealSense camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        print("Failed to grab frame")
        continue

    # Convert RealSense frame to numpy arrays
    frame = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    height, width = frame.shape[:2]

    # Select bounding box if not done
    if not enter_pressed:
        temp_frame = frame.copy()
        if drawing and ix >= 0 and iy >= 0:  # While drawing
            cv2.rectangle(frame, (ix, iy), (fx, fy), (255, 0, 0), 2)        
        cv2.putText(frame, "Select an object by drawing a box", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or Q to quit
            break
    else:
        if not if_init:
            if_init = True
            predictor.load_first_frame(frame)
            using_box = True 
            ann_frame_idx = 0
            ann_obj_id = (1)
            labels = np.array([1], dtype=np.int32)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
                )
        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)

            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            # Process the output mask
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).byte().cuda()
                all_mask = out_mask.cpu().numpy() * 255

            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
            
        # Show the frame
        cv2.imshow("Camera", frame)

        # Break if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
