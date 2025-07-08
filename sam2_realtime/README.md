- Camera rate, i.e. camera must have no lag. Test different resolutions;
- Test different ways of computing the center of the mask;
- Confirm the center is computed correctly (Yolo vs tiago obj local);
- Optimize Kalman Filter;
- Confirm TF publication;


# Depth info
Realsense
depth_scale = 1000

Orbbec
depth_scale = 1

Microsoft Azure
depth_scale = 1


# TODO
- Test LifeCycle properties;
- Update and test Docker;
- Apply transform to relevant frame (frame_id: rgb_camera_link). Fix y=0;
- Add event_in to tracker;
- Add time-gated outlier rejection;
