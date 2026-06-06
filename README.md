# semantic_spatial_mapping

Start here, then scroll to **05.06.26 Integration Report** below. That entry records the baseline repository state and the v2.0 integration changes.

## Current Purpose

`semantic_spatial_mapping` is a robotics perception and VSLAM integration stack. It combines RGB-D geometry, semantic segmentation, semantic object mapping, configurable pose sources, and visual odometry behind a ROS2 deployment runtime.

The current target is a plug-and-play, hardware/software-agnostic perception + VSLAM stack that can run in:

- Gazebo simulation trials. See `gazebo_demo/README.md` for the factory world, bridge, perception/VSLAM demo, and exact terminal commands.
- embedded robot hardware
- ROS2 bag/replay workflows
- headless validation runs

## Current Runtime Shape

```text
RGB image + depth image + CameraInfo
    -> segmentation backend
    -> mask quality filtering
    -> RGB-D semantic point extraction
    -> semantic object/entity fusion
    -> pose source or internal VSLAM
    -> world-frame map update
    -> ROS diagnostics, clouds, objects, odometry, runtime report
```

## Repository Layout

```text
geometry/                    RGB-D geometry, camera models, transforms, point clouds
mapping/global_map/           bounded world map and semantic object map
motion/vo/                    visual odometry frontend and VSLAM backend scaffolding
runtime/core/                 deployment runtime, config, diagnostics, logging, providers
ros/semantic_spatial_mapping_ros/
                              ROS2 package, launch files, node, publishers, converters
segmentation/                 YOLO reference segmentation backend
tracking/                     temporal filtering and tracking references
notes/                        setup notes and runtime validation checklist
tests/                        no-runtime-data smoke and failure-mode tests
requirements/                 Python/runtime dependency lists
```

## Install Dependencies

Create and activate a Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
```

`requirements/python_requirements.txt` is currently UTF-16 LE. Convert it to a temporary UTF-8 requirements file before installing:

```bash
iconv -f UTF-16LE -t UTF-8 requirements/python_requirements.txt > /tmp/ssm_python_requirements.txt
pip install -r /tmp/ssm_python_requirements.txt
```

For ROS2 runtime usage, install/source a ROS2 distribution with at least:

```text
rclpy
sensor_msgs
nav_msgs
geometry_msgs
std_msgs
tf2_ros
launch
launch_ros
rosbag2
```

The YOLO backend requires `ultralytics`, `torch`, and `torchvision`. For headless smoke tests or Gazebo with segmentation disabled, those heavy dependencies are not imported.

## Local Validation

Run these before a ROS2 or hardware session:

```bash
python3 -m compileall runtime/core mapping/global_map motion/vo segmentation ros/semantic_spatial_mapping_ros/semantic_spatial_mapping_ros tests
python3 -m unittest tests.test_runtime_core tests.test_ros_converters tests.test_vo_rgbd
python3 -m runtime.core.runtime_validation
python3 ros/semantic_spatial_mapping_ros/setup.py --name
```

Expected:

```text
OK
runtime validation passed
semantic_spatial_mapping_ros
```

## ROS2 Build And Launch

From a ROS2 workspace, place or symlink the package so the ROS package path is available to `colcon`. One workable layout is:

```text
<workspace>/src/semantic_spatial_mapping/ros/semantic_spatial_mapping_ros
```

Build:

```bash
cd <workspace>
colcon build --packages-select semantic_spatial_mapping_ros
source install/setup.bash
```

Run Gazebo profile for the perception/VSLAM ROS node:

```bash
ros2 launch semantic_spatial_mapping_ros gazebo_runtime.launch.py
```

For the full Gazebo trial, including the factory world, `factory_bot`, ROS-Gazebo bridge, reactive navigator, and screen-recording command sequence, use [`gazebo_demo/README.md`](gazebo_demo/README.md).

Run embedded OAK-D style profile:

```bash
ros2 launch semantic_spatial_mapping_ros embedded_runtime.launch.py
```

Override config:

```bash
ros2 launch semantic_spatial_mapping_ros gazebo_runtime.launch.py \
  config_path:=/absolute/path/to/runtime.yaml
```

Default configs:

```text
ros/semantic_spatial_mapping_ros/config/gazebo.yaml
ros/semantic_spatial_mapping_ros/config/embedded_oakd.yaml
```

Runtime checklist:

```text
notes/runtime_validation_checklist.md
```

## Runtime Logging

Every ROS2 stack run creates a text report:

```text
runtime_logs/<timestamp>_<profile>_<pid>/runtime_report.txt
```

Embedded profiles also try to record a rosbag:

```text
runtime_logs/<timestamp>_<profile>_<pid>/rosbag/
```

`runtime_logs/` is intentionally not in `.gitignore`. These reports are meant to preserve field-run evidence when debugging sensor topics, encodings, CameraInfo, TF, calibration, perception, mapping, or VSLAM behavior.

Runtime logging config:

```yaml
logging:
  enabled: true
  directory: runtime_logs
  frame_log_period: 1
  record_embedded_rosbag: true
  rosbag_topics: []
```

The report includes the resolved config, topic list, image/depth message metadata, depth validity ratios, CameraInfo intrinsics, pose status, diagnostics, semantic object counts, exceptions, and shutdown summary.

## ROS2 Topics

Default input topics are configurable in YAML:

```text
RGB image:        /camera/color/image_raw or /vctr/rgb_raw
Depth image:      /camera/depth/image_raw or /vctr/depth_raw
RGB CameraInfo:   /camera/color/camera_info or /vctr/rgb/camera_info
Depth CameraInfo: /camera/depth/camera_info or /vctr/depth/camera_info
Pose source:      /odom, /pose, TF, internal VSLAM, or identity fallback
```

Default output topics:

```text
/semantic_spatial/points
/semantic_spatial/objects
/semantic_spatial/map
/semantic_spatial/diagnostics
/semantic_spatial/visual_odometry
/semantic_spatial/debug_overlay
```

## Pose Contract

The runtime consumes camera pose as:

```text
T_world_cam
```

ROS pose inputs declare their contract:

```yaml
pose:
  input_pose: world_camera
```

or:

```yaml
pose:
  input_pose: world_base
```

For `world_base`, the runtime computes:

```text
T_world_cam = T_world_base @ T_base_camera
```

using:

```yaml
extrinsics:
  base_to_camera:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
```

Replace identity extrinsics with measured robot calibration before trusting map geometry.

## Current Capabilities

| Area | Current state |
| --- | --- |
| RGB-D geometry | Vectorized point extraction, depth units, projection helpers |
| Runtime core | Config-driven deployment runtime with diagnostics and degraded modes |
| ROS2 package | Launch files, node, converters, sync, pose adapters, publishers |
| Pose sources | Odometry, PoseStamped, TF, internal VSLAM, identity fallback |
| Perception | Backend interface with disabled, mock, and YOLO providers |
| Mask handling | Area, border-touch, depth-support, and max-mask filters |
| Semantic map | Bounded point map plus object/entity map with IDs and observations |
| VSLAM | RGB-D PnP path with depth landmarks and monocular fallback |
| Logging | Per-run text reports; embedded rosbag recording attempt |
| Tests | Core runtime, ROS converters, failure modes, RGB-D VO PnP |

## Known Gaps Before Calling It Finished

- Needs `colcon build` and ROS launch validation in a sourced ROS2 workspace.
- Needs Gazebo, rosbag, and embedded hardware runtime feedback.
- Camera optical frame, base-to-camera, and depth-to-RGB extrinsics need real calibration checks.
- QoS and sync behavior may need sensor-specific tuning.
- Internal VSLAM still needs robust local BA, relocalization, loop validation, and pose graph correction.
- Semantic object fusion is currently nearest-class/centroid based; it needs runtime evaluation.
- Runtime performance needs profiling on embedded hardware.

## 05.06.26 Integration Report

### Starting Point

Before the v2.0 integration pass, the repository already had modular research/reference components:

- `geometry/` for RGB-D backprojection, transforms, point clouds, and OBB support.
- `segmentation/` with a YOLOv8 segmentation reference.
- `motion/vo/` with a monocular visual odometry frontend.
- `mapping/global_map/` with world-frame point accumulation.
- `runtime/python_reference/` with a Python reference semantic runtime.
- `tracking/` with temporal filtering and state-estimation references.
- `ingestion/` and replay utilities for dataset/rosbag-style data paths.

At that stage, the repo was organized around modular perception and VO prototypes. The ROS deployment layer was not yet a working runtime package, pose-source selection was not generalized, world-map storage was append-oriented, RGB-D extraction used more prototype-style loops, and the VO backend pieces from the external VSLAM reference had not been integrated into this repo's `motion/vo` package.

### Changes Added In v2.0

Runtime core:

- Added `runtime/core/` as the deployment-facing runtime layer.
- Added structured runtime config loading and validation.
- Added diagnostics, health state, degraded modes, pose providers, runtime output types, and lifecycle hooks.
- Added runtime logger that writes `runtime_logs/<session>/runtime_report.txt` on every ROS2 run.
- Added embedded-only rosbag recording attempt through `ros2 bag record`.

ROS2 deployment:

- Added `ros/semantic_spatial_mapping_ros/` package.
- Added Gazebo and embedded OAK-D style YAML profiles.
- Added launch files for both profiles.
- Added `semantic_spatial_node`.
- Added ROS image/CameraInfo conversion without requiring `cv_bridge`.
- Added approximate RGB-D sync.
- Added ROS pose adapters for Odometry, PoseStamped, TF, internal VSLAM, and identity fallback.
- Added publishers for semantic points, semantic objects, map points, diagnostics, VO odometry, and optional overlay.
- Added sensor-data QoS for camera streams.

Geometry and mapping:

- Vectorized point cloud generation and object point extraction.
- Added configurable depth units.
- Added depth-to-RGB point projection helpers.
- Replaced append-only world map behavior with bounded voxel-downsampled storage.
- Added semantic object map with IDs, labels, confidence, centroid, extent, covariance, observation count, and bounded fused points.

Perception:

- Added perception contracts: `InstanceMask`, `SemanticFrame`, and `ObjectGeometry`.
- Added segmentation provider abstraction: disabled, mock, YOLO.
- Preserved YOLO class IDs, labels, confidences, and boxes.
- Added mask filtering by area, depth support, border contact, and max masks per frame.
- Added point outlier filtering before map/object fusion.

VSLAM and VO:

- Integrated external VSLAM backend modules into `motion/vo/`.
- Preserved the existing `VisualOdometry.update(...) -> PoseUpdate` API used by runtime code.
- Added covisibility graph plumbing.
- Added RGB-D scale handling.
- Added RGB-D PnP tracking path using keyframe depth and current 2D features.
- Added depth-initialized RGB-D landmarks when PnP succeeds.
- Kept monocular essential-matrix tracking as fallback.
- Added VO diagnostics for tracking method, depth support ratio, and reprojection error.
- Made optional visualization imports lazy.

Reliability and debugging:

- Added config validation for topics, frames, depth ranges, pose source, pose contract, extrinsics, resource limits, and logging settings.
- Added external pose staleness checks.
- Added runtime pose guards for NaN/non-finite poses, huge translations, and sudden jumps.
- Added CameraInfo staleness filtering.
- Added resource caps for masks, per-frame points, and per-object points.
- Added runtime validation checklist in `notes/runtime_validation_checklist.md`.

Tests:

- Added synthetic runtime validation.
- Added unit tests for config validation, stale pose, missing depth, missing CameraInfo, empty segmentation, bad extrinsics, NaN pose, pose jumps, object fusion, point caps, runtime logging, and default rosbag topics.
- Added ROS converter tests that run when ROS message packages are available.
- Added RGB-D VO PnP test.

### Verification Run

The following local checks passed:

```bash
python3 -m compileall runtime/core mapping/global_map motion/vo segmentation ros/semantic_spatial_mapping_ros/semantic_spatial_mapping_ros tests
python3 -m unittest tests.test_runtime_core tests.test_ros_converters tests.test_vo_rgbd
python3 -m runtime.core.runtime_validation
python3 ros/semantic_spatial_mapping_ros/setup.py --name
git check-ignore -v runtime_logs || true
```

The unit test suite currently reports:

```text
20 tests OK
```

### Current Interpretation

The v2.0 integration moved the repository from a modular perception/VO reference codebase into a first deployable architecture pass for a hardware/software-agnostic perception + VSLAM stack. The next milestone is runtime validation with Gazebo, rosbag replay, and embedded hardware so the stack can be hardened against real TF trees, topic QoS, depth encodings, CameraInfo timing, calibration, object fusion behavior, and VSLAM tracking quality.

## Historical Snapshot: Earlier README Direction

The earlier README described the repository as a modular semantic spatial mapping framework integrating:

- RGB-D geometry
- semantic segmentation
- persistent object tracking
- monocular visual odometry
- world-frame spatial accumulation
- semantic point cloud extraction

The earlier pipeline was:

```text
RGB Frame
Depth Frame
    -> Segmentation
    -> Semantic Masks
    -> RGB-D Pointcloud Extraction
    -> Visual Odometry
    -> World-Frame Projection
    -> Persistent Semantic Spatial Map
```

The earlier roadmap focused on:

- static scene stability
- rotation and translation consistency
- semantic persistence
- Open3D visualization
- RGB-D metric scale grounding
- persistent semantic entities
- relocalization
- local semantic mapping
- pose graph optimization
- loop closure
- navigation/manipulation integration

That direction remains the project direction. The current README records the deployment runtime, ROS2 integration, logging, testing, and object/VSLAM upgrades added on `05.06.26`.
