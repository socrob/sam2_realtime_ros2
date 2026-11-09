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
