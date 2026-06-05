# Runtime Validation Checklist

Use this when running Gazebo, rosbag replay, or embedded hardware.

## Before Launch

- Confirm the selected config:
  - `ros/semantic_spatial_mapping_ros/config/gazebo.yaml`
  - `ros/semantic_spatial_mapping_ros/config/embedded_oakd.yaml`
- Confirm RGB topic publishes `sensor_msgs/Image`.
- Confirm depth topic publishes `sensor_msgs/Image`.
- Confirm RGB and depth `CameraInfo` topics publish valid `K` matrices.
- Confirm camera topics are compatible with ROS sensor-data QoS.
- Confirm depth encoding:
  - Gazebo commonly uses `32FC1` meters, so `depth.depth_unit: 1.0`.
  - OAK-D style streams commonly use `16UC1` millimeters, so `depth.depth_unit: 0.001`.

## Pose Contract

The runtime always consumes camera pose as:

```text
T_world_cam
```

ROS pose inputs must declare their contract in config:

```yaml
pose:
  input_pose: world_camera
```

or:

```yaml
pose:
  input_pose: world_base
```

When `world_base` is used, the runtime computes:

```text
T_world_cam = T_world_base @ T_base_camera
```

and requires:

```yaml
extrinsics:
  base_to_camera:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
```

Replace the identity matrix with the calibrated camera pose in the robot base
frame before trusting map geometry.

## Launch

Gazebo:

```bash
ros2 launch semantic_spatial_mapping_ros gazebo_runtime.launch.py
```

Embedded:

```bash
ros2 launch semantic_spatial_mapping_ros embedded_runtime.launch.py
```

Override config:

```bash
ros2 launch semantic_spatial_mapping_ros gazebo_runtime.launch.py \
  config_path:=/absolute/path/to/runtime.yaml
```

## Runtime Report

Every run writes a text report:

```text
runtime_logs/<timestamp>_<profile>_<pid>/runtime_report.txt
```

The report includes:

- full resolved runtime config
- configured topics
- platform, process, and launch context
- frame-level RGB image shape, dtype, encoding, frame ID, timestamp, and byte size
- frame-level depth image shape, dtype, encoding, valid-depth ratio, nonzero ratio, and depth unit
- CameraInfo frame IDs, timestamps, dimensions, intrinsics, and distortion metadata
- pose source, status, frame IDs, translation, and finite-matrix checks
- diagnostics, object counts, map outputs, dropped frames, sync failures, stale CameraInfo, and exceptions

Embedded profiles also try to record a rosbag:

```text
runtime_logs/<timestamp>_embedded_oakd_<pid>/rosbag/
```

The default embedded bag topics include RGB, depth, CameraInfo, pose source
topic when applicable, semantic outputs, diagnostics, and VO odometry. Add more
topics through:

```yaml
logging:
  rosbag_topics:
    - /extra/topic
```

If rosbag recording cannot start, the text report records the reason, for
example `ros2_cli_not_found`.

## First Diagnostics To Inspect

Monitor:

```bash
ros2 topic echo /semantic_spatial/diagnostics
```

Expected healthy fields:

- `health: OK`
- `rgb_received: true`
- `depth_received: true`
- `rgb_camera_info_received: true`
- `depth_camera_info_received: true`
- `camera_info_ready: true`
- `pose_source_active: true`
- `tracking_ok: true`
- `pose_age_sec` below `pose.max_age_sec`
- `camera_info_stale: false`
- `masks_processed` below `segmentation.max_masks_per_frame`
- `points_generated` below `mapping.max_points_per_frame`
- `semantic_objects` increases when stable objects are observed
- `sync_failures` not steadily increasing
- `dropped_frames` not steadily increasing
- `last_error` empty

## If Geometry Looks Wrong

Check in this order:

1. Depth unit and encoding.
2. RGB and depth CameraInfo intrinsics.
3. `extrinsics.depth_to_rgb`.
4. `pose.input_pose`.
5. `extrinsics.base_to_camera`.
6. TF tree timestamps and frame IDs.
7. Camera optical frame convention.
8. Pose guard diagnostics such as `INVALID_POSE_JUMP` or `INVALID_POSE_NONFINITE`.

## Resource Limits

The runtime has conservative guards for bad frames:

```yaml
pose:
  max_age_sec: 0.25
  max_jump_m: 5.0
  max_translation_norm_m: 10000.0

segmentation:
  max_masks_per_frame: 8

mapping:
  max_points_per_frame: 20000
  max_points_per_object: 5000
  max_objects: 512
  object_association_distance_m: 0.75
```

Tune these only after looking at bag or hardware diagnostics.

## VSLAM Algorithm Notes

When depth is available for matched keyframe features, internal VSLAM now tries
RGB-D PnP before monocular essential-matrix tracking:

```text
keyframe feature + keyframe depth -> 3D point in keyframe camera
current feature pixel             -> 2D observation
solvePnPRansac                    -> metric T_cur_ref
```

If depth support is insufficient, it falls back to monocular essential-matrix
tracking and reports degraded scale quality through the normal VO diagnostics.

## Local No-ROS Smoke Tests

Run:

```bash
python3 -m runtime.core.runtime_validation
python3 -m unittest tests.test_runtime_core tests.test_ros_converters
```

Both should pass before a runtime debugging session.
