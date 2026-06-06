# Gazebo Demo

This folder contains a self-contained Gazebo Sim factory obstacle world and a live factory_bot navigation demo wired into the perception/VSLAM runtime topics.

World file:

```text
gazebo_demo/factory_obstacle_demo.world
```

The world contains a warehouse/factory floor, boundary walls, racks, pallets, crates, barrels, safety posts, a marked start zone, a visible green arrival target, and `factory_bot`, a boxy tracked robot used for internal demo runs.

The demo intentionally does not preload obstacle locations into the navigator. The robot receives a target direction/displacement, consumes live RGB, depth, odom, and semantic stack diagnostics, then adjusts speed and steering from what the sensors report.

## Runtime Commands

Use four terminals for the full demo. Keep Terminal 4 visible during screen recording if you want perception/navigation logs on screen.

Terminal 1: open Gazebo Sim with the factory world.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
gz sim gazebo_demo/factory_obstacle_demo.world
```

Press Play in the Gazebo GUI if the simulation opens paused.

Terminal 2: bridge Gazebo camera, depth, odom, and command topics into ROS.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
ros2 launch gazebo_demo/launch/factory_bot_bridge.launch.py
```

Terminal 3: run the perception/VSLAM stack against the bridged Gazebo topics.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
source install/setup.bash
ros2 launch semantic_spatial_mapping_ros gazebo_runtime.launch.py
```

If the ROS workspace has not been built yet:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
colcon build
source install/setup.bash
```

Terminal 4: run the reactive navigator.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/neel-mukherjee/Desktop/semantic_spatial_mapping
python3 gazebo_demo/scripts/factory_bot_stack_navigator.py
```

The navigator logs live perception state, including depth sectors, mode, distance to goal, heading error, command velocity, pose source, target marker visibility, stack diagnostics, and semantic object counts.

## Topic Contract

Gazebo publishes scoped transport topics; `factory_bot_bridge.launch.py` maps them to the stable ROS topics used by the stack.

ROS topics used by the perception/VSLAM stack:

- `/camera/color/image_raw`
- `/camera/color/camera_info`
- `/camera/depth/image_raw`
- `/camera/depth/camera_info`
- `/odom`
- `/semantic_spatial/diagnostics`
- `/semantic_spatial/objects`
- `/semantic_spatial/visual_odometry`

Command path:

- navigator publishes ROS `geometry_msgs/msg/Twist` on `/factory_bot/cmd_vel`
- bridge forwards it to Gazebo `/model/factory_bot/cmd_vel`
- Gazebo DiffDrive publishes `/model/factory_bot/odometry`, bridged back to ROS `/odom`

## Manual Checks

List Gazebo topics:

```bash
gz topic -l | grep factory_bot
```

List ROS topics after the bridge is running:

```bash
ros2 topic list
```

Stop the robot manually:

```bash
ros2 topic pub --once /factory_bot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"
```

Send a manual forward command for actuator testing only:

```bash
ros2 topic pub /factory_bot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.25}, angular: {z: 0.0}}" -r 10
```

Stop it with Ctrl-C, then send the zero command above.

## Notes

- `factory_bot_demo_driver.py` is kept as a simple actuator smoke test, but the intended demo path is `factory_bot_stack_navigator.py` with the ROS bridge and perception/VSLAM stack running.
- The target used by the navigator is an odom-frame displacement from the spawn area to the green arrival target across the warehouse.
- The stack logs still provide the deeper runtime report files under `runtime_logs/` when the ROS node is running.
