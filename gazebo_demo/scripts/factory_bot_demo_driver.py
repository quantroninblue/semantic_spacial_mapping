#!/usr/bin/env python3
"""Gazebo Sim demo driver for factory_bot.

Publishes directly to Gazebo Transport so the robot moves in `gz sim` without a
ROS-Gazebo bridge. Stop with Ctrl-C; it sends a final zero velocity command.
"""

from __future__ import annotations

import shutil
import subprocess
import time

TOPIC = "/model/factory_bot/cmd_vel"
MSG_TYPE = "gz.msgs.Twist"


def command_for_phase(phase: float) -> tuple[float, float]:
    if phase < 7.0:
        return 0.22, 0.0
    if phase < 14.0:
        return 0.18, 0.32
    if phase < 20.0:
        return 0.20, -0.22
    if phase < 25.0:
        return 0.14, 0.42
    if phase < 29.0:
        return 0.18, 0.0
    return 0.0, 0.55


def publish(linear_x: float, angular_z: float) -> None:
    payload = f"linear: {{x: {linear_x:.3f}}}, angular: {{z: {angular_z:.3f}}}"
    try:
        subprocess.run(
            ["gz", "topic", "-t", TOPIC, "-m", MSG_TYPE, "-p", payload],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        )
    except subprocess.TimeoutExpired:
        pass


def main() -> None:
    if shutil.which("gz") is None:
        raise SystemExit("gz command not found. Source ROS/Gazebo first: source /opt/ros/jazzy/setup.bash")

    print(f"Publishing Gazebo Sim velocity commands to {TOPIC}", flush=True)
    print("Open gz sim, press Play, then leave this running. Ctrl-C stops the robot.", flush=True)
    start = time.monotonic()
    try:
        while True:
            phase = (time.monotonic() - start) % 32.0
            linear_x, angular_z = command_for_phase(phase)
            publish(linear_x, angular_z)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping factory_bot", flush=True)
    finally:
        for _ in range(5):
            publish(0.0, 0.0)
            time.sleep(0.05)


if __name__ == "__main__":
    main()
