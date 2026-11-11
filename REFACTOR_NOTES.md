# Refactor notes (`perception-refactor` branch)

## Default asset folders (2026-11-09)

Every laptop used to need `SAM2_ASSETS_DIR` and `YOLO_ASSETS_DIR` exported (by `setup_env.sh` into `~/.bashrc`, pointing into the git submodule). Both variables are now optional overrides of fixed default folders:

| Variable | Default | Holds |
|---|---|---|
| `SAM2_ASSETS_DIR` | `~/sam2_assets` | `checkpoints/sam2.1_*.pt` |
| `YOLO_ASSETS_DIR` | `~/yolo_ros/models` | YOLO weights for the prompt nodes |

- Launch scripts (`sam2_realtime_node.sh`, `sam2_realtime_multi_node.sh`, `master.sh`, `sam2_tracking_pipeline.sh`) default to `~/sam2_assets` and `cd` there, so the relative `checkpoints/...` paths keep working. They now exit with a message pointing at `setup_env.sh` when the folder or checkpoint is missing. Before, `master.sh` and `sam2_tracking_pipeline.sh` skipped the `cd` when the variable was unset and SAM2 failed later on the checkpoint load.
- `yolo_prompt_node.py` / `yolo_mask_prompt_node.py` default to `~/yolo_ros/models` instead of raising; `yolo_prompt_node.sh` no longer exits when the variable is unset.
- `setup_env.sh` can be run from any directory. It downloads the checkpoints into `<assets>/checkpoints` by running the submodule's `download_ckpts.sh` from there (the submodule itself is unchanged), skips the download when all four are present (wget would otherwise save `*.pt.1` copies), and no longer edits `~/.bashrc`.
- The launch files (`*.launch.py`) still take `checkpoint` relative to the working directory; run `ros2 launch` from the assets folder, or pass an absolute path.
- Existing checkouts whose checkpoints are inside the submodule: make `~/sam2_assets` a symlink to `sam2_realtime/sam2_realtime/segment-anything-2-real-time` (or move the `checkpoints/*.pt` files into `~/sam2_assets/checkpoints`).
- The Docker image still sets `SAM2_ASSETS_DIR` to the in-container submodule path; it remains a valid override.

## YOLO prompt nodes, extra nodes and `event_in` removed; one launch file (2026-11-11)

The package only tracks: the prompt comes from an external detector, as a `PromptBbox` on `/sam2/init_prompt` or a mask image on `/sam2/init_prompt_mask` (both inputs unchanged).

- **Removed nodes:** `yolo_prompt_node` and `yolo_mask_prompt_node` (their own YOLO models), `bbox_prompt_node` (a mouse-drawn box in an OpenCV window), `sam2_realtime_multi_node` (an older copy of `sam2_realtime_node`) and `debug_node` (unlaunched; `sam2_realtime_node` publishes `img_mask`). `setup.py` keeps `sam2_realtime_node`, `mask2pcl` and `track_node`. `ultralytics` and `lap` (only used by them) are gone from `requirements.txt`.
- **`event_in` and `enable` removed from `track_node` and `mask2pcl`:** they process whenever they are ACTIVE; pause with a lifecycle deactivate, resume with activate. `on_activate` resets what `e_start` reset (measurement, centroid, update time); `track_node`'s `on_deactivate` zeroes the EKF velocity, as `e_stop` did, so the predicted TF doesn't drift while paused.
- **Launch files:** only `sam2_tracking_pipeline.launch.py` (its `launch_sam2`/`launch_mask2pcl`/`launch_track_node` flags cover the single nodes) and `sam2_tracking_pipeline.sh`. Removed: `master.launch.py`/`.sh`, the per-node `*.launch.py` and `.sh` files, and the prompt, multi and debug scripts.
  - `sam2_tracking_pipeline.launch.py`: the arguments come from one table grouped by node; `mask2pcl_enable` and `track_enable` are gone; `fix_height`/`fixed_height` are listed with the tracker (they were under Mask2PCL). Same nodes, parameters and defaults otherwise.
  - `sam2_tracking_pipeline.sh`: `--camera azure|realsense`, node selection (`--all`, `--tracking`, `--pcl`, `--sam2-only`), `--target-frame` (default `map`, as before), `--live-viz`, `--venv`, and any other launch argument after `--`. The other parameters take the launch file's defaults instead of repeating them. Kept from before: live visualization on, `fix_height` true at 1.20 m. Changed: the RealSense uses `aligned_depth_to_color` (driver with `align_depth.enable:=true`); the unused camera-default target frame logic (it was always overridden to `map`) and the unsupported `sim_head`/`sim_wrist` names are gone.
- **For clients:** anything that published `e_start`/`e_stop` on `/track_node/event_in` or `/mask2pcl/event_in`, or passed `track_enable`/`mask2pcl_enable` to the launch file, must drop it (the events now do nothing; unknown launch arguments are ignored).
- README: structure, how it works and the run example rewritten for the pipeline launch and an external prompt.
- **Checked live** (2026-11-11, RealSense D435, no robot): SAM2 prompted with a person from an external YOLO detection → masks at 15 Hz and `tracked_object` at 10 Hz with no `e_start`; `ros2 lifecycle set /track_node/track_node deactivate` → `tracked_object` stops, `activate` → back at 10 Hz. `sam2_tracking_pipeline.sh --camera realsense --all --target-frame camera_link` run from another directory → SAM2, Mask2PCL (`object_cloud` 15 Hz) and tracker (10 Hz). Passing the old `track_enable`/`mask2pcl_enable` to the launch file gives no error. Clean install: only `sam2_realtime_node`, `mask2pcl`, `track_node` and the pipeline launch file and script.

## GitHub workflow removed (2026-11-11)

- `.github/workflows/ci.yml` (build plus the ament linters) is removed. It had failed on every `main` run since May on the linters (missing copyright headers, about 570 flake8 warnings), so it said nothing about the changes.
