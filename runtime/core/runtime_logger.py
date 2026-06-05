from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import RuntimeConfig


class RuntimeSessionLogger:
    def __init__(
        self,
        config: RuntimeConfig,
        config_path: str = "",
    ):
        self.config = config
        self.config_path = config_path
        self.enabled = bool(config.logging.enabled)
        self.session_id = self._session_id(config.profile)
        self.session_dir = Path(config.logging.directory) / self.session_id
        self.report_path = self.session_dir / "runtime_report.txt"
        self.rosbag_dir = self.session_dir / "rosbag"
        self._file = None
        self._rosbag_process: Optional[subprocess.Popen] = None
        self._frame_count = 0
        self._error_count = 0
        self._start_time = time.time()

        if self.enabled:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(self.report_path, "a", encoding="utf-8")
            self._write_header()

    @property
    def rosbag_active(self) -> bool:
        return self._rosbag_process is not None and self._rosbag_process.poll() is None

    def log_event(self, event: str, **fields) -> None:
        self._write_record("EVENT", {"event": event, **fields})

    def log_subscriptions(self, topics: dict[str, str]) -> None:
        self._write_section("CONFIGURED_TOPICS")
        for name, topic in sorted(topics.items()):
            self._write_line(f"{name}: {topic}")
        self._write_line("")

    def start_embedded_rosbag(self, topics: list[str]) -> None:
        if not self.enabled:
            return

        if "embedded" not in (self.config.profile or "").lower():
            self.log_event(
                "rosbag_not_started",
                reason="profile_not_embedded",
                profile=self.config.profile,
            )
            return

        if not self.config.logging.record_embedded_rosbag:
            self.log_event("rosbag_not_started", reason="disabled_by_config")
            return

        ros2 = shutil.which("ros2")
        if ros2 is None:
            self.log_event("rosbag_start_failed", reason="ros2_cli_not_found")
            return

        topics = [topic for topic in topics if topic]
        if not topics:
            self.log_event("rosbag_start_failed", reason="no_topics")
            return

        cmd = [
            ros2,
            "bag",
            "record",
            "-o",
            str(self.rosbag_dir),
            *topics,
        ]
        try:
            self._rosbag_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            self.log_event("rosbag_start_failed", reason=str(exc), command=cmd)
            return

        self.log_event(
            "rosbag_started",
            command=cmd,
            output_dir=str(self.rosbag_dir),
            topics=topics,
        )

    def log_frame(
        self,
        frame_packet,
        output,
        rgb_msg=None,
        depth_msg=None,
        rgb_info=None,
        depth_info=None,
    ) -> None:
        if not self.enabled:
            return

        self._frame_count += 1
        period = max(int(self.config.logging.frame_log_period), 1)
        should_log = self._frame_count == 1 or self._frame_count % period == 0
        diagnostics = output.diagnostics or {}
        if diagnostics.get("health") != "OK" or diagnostics.get("last_error"):
            should_log = True
        if not should_log:
            return

        record = {
            "frame_id": frame_packet.frame_id,
            "timestamp": frame_packet.timestamp,
            "source": frame_packet.source,
            "rgb": self._array_report(frame_packet.rgb_frame),
            "depth": self._depth_report(frame_packet.depth_frame),
            "rgb_msg": self._image_msg_report(rgb_msg),
            "depth_msg": self._image_msg_report(depth_msg),
            "rgb_camera_info": self._camera_info_report(rgb_info),
            "depth_camera_info": self._camera_info_report(depth_info),
            "pose": self._pose_report(output.pose),
            "world_point_batches": len(output.world_points),
            "semantic_objects": len(output.semantic_objects),
            "diagnostics": diagnostics,
        }
        self._write_record("FRAME", record)

    def log_exception(self, where: str, exc: BaseException, traceback_text: str = "") -> None:
        self._error_count += 1
        self._write_record(
            "EXCEPTION",
            {
                "where": where,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback_text,
            },
        )

    def close(self) -> None:
        if self._rosbag_process is not None:
            self._stop_rosbag()

        elapsed = time.time() - self._start_time
        self._write_section("SUMMARY")
        self._write_line(f"frames_logged_or_seen: {self._frame_count}")
        self._write_line(f"errors: {self._error_count}")
        self._write_line(f"elapsed_sec: {elapsed:.3f}")
        self._write_line(f"closed_utc: {datetime.now(timezone.utc).isoformat()}")
        self._write_line("")

        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def _stop_rosbag(self) -> None:
        proc = self._rosbag_process
        if proc is None:
            return
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                output, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                output, _ = proc.communicate(timeout=5)
        else:
            output, _ = proc.communicate(timeout=1)

        self.log_event(
            "rosbag_stopped",
            returncode=proc.returncode,
            output_tail=(output or "")[-4000:],
            output_dir=str(self.rosbag_dir),
        )
        self._rosbag_process = None

    def _write_header(self) -> None:
        self._write_section("RUNTIME_SESSION")
        self._write_line(f"session_id: {self.session_id}")
        self._write_line(f"profile: {self.config.profile}")
        self._write_line(f"config_path: {self.config_path}")
        self._write_line(f"report_path: {self.report_path}")
        self._write_line(f"created_utc: {datetime.now(timezone.utc).isoformat()}")
        self._write_line(f"pid: {os.getpid()}")
        self._write_line(f"cwd: {os.getcwd()}")
        self._write_line(f"python: {sys.version.replace(chr(10), ' ')}")
        self._write_line(f"platform: {platform.platform()}")
        self._write_line("")
        self._write_section("RUNTIME_CONFIG_JSON")
        self._write_line(json.dumps(_jsonable(asdict(self.config)), indent=2, sort_keys=True))
        self._write_line("")

    def _write_section(self, name: str) -> None:
        self._write_line(f"===== {name} =====")

    def _write_record(self, kind: str, payload: dict[str, Any]) -> None:
        self._write_line(
            json.dumps(
                {
                    "kind": kind,
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    **_jsonable(payload),
                },
                sort_keys=True,
            )
        )

    def _write_line(self, line: str) -> None:
        if not self.enabled or self._file is None:
            return
        self._file.write(line + "\n")
        self._file.flush()

    @staticmethod
    def _session_id(profile: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_profile = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in (profile or "runtime")
        )
        return f"{stamp}_{safe_profile}_{os.getpid()}"

    @staticmethod
    def _array_report(array) -> dict:
        if array is None:
            return {"present": False}
        arr = np.asarray(array)
        report = {
            "present": True,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        if arr.size:
            finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else arr
            if np.size(finite):
                report.update(
                    {
                        "min": float(np.min(finite)),
                        "max": float(np.max(finite)),
                    }
                )
        return report

    def _depth_report(self, depth) -> dict:
        report = self._array_report(depth)
        if depth is None:
            return report
        arr = np.asarray(depth)
        depth_m = arr.astype(np.float32) * float(self.config.depth.depth_unit)
        valid = (
            np.isfinite(depth_m)
            & (depth_m > 0.0)
            & (depth_m >= self.config.depth.min_m)
            & (depth_m <= self.config.depth.max_m)
        )
        report.update(
            {
                "depth_unit": self.config.depth.depth_unit,
                "valid_ratio": (
                    float(np.count_nonzero(valid)) / float(valid.size)
                    if valid.size
                    else 0.0
                ),
                "nonzero_ratio": (
                    float(np.count_nonzero(arr)) / float(arr.size)
                    if arr.size
                    else 0.0
                ),
            }
        )
        return report

    @staticmethod
    def _image_msg_report(msg) -> dict:
        if msg is None:
            return {"present": False}
        return {
            "present": True,
            "frame_id": msg.header.frame_id,
            "stamp_sec": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
            "height": int(msg.height),
            "width": int(msg.width),
            "encoding": msg.encoding,
            "step": int(msg.step),
            "data_bytes": len(msg.data),
        }

    @staticmethod
    def _camera_info_report(msg) -> dict:
        if msg is None:
            return {"present": False}
        return {
            "present": True,
            "frame_id": msg.header.frame_id,
            "stamp_sec": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
            "height": int(msg.height),
            "width": int(msg.width),
            "k": [float(v) for v in msg.k],
            "d_len": len(msg.d),
            "distortion_model": msg.distortion_model,
        }

    @staticmethod
    def _pose_report(pose) -> dict:
        if pose is None:
            return {"present": False}
        T = np.asarray(pose.T_world_cam)
        return {
            "present": True,
            "success": bool(pose.success),
            "source": pose.source,
            "status": pose.status,
            "frame_id": pose.frame_id,
            "child_frame_id": pose.child_frame_id,
            "translation": T[:3, 3].tolist() if T.shape == (4, 4) else [],
            "finite": bool(np.isfinite(T).all()) if T.shape == (4, 4) else False,
        }


def default_rosbag_topics(config: RuntimeConfig) -> list[str]:
    topics = [
        config.topics.rgb,
        config.topics.depth,
        config.topics.rgb_camera_info,
        config.topics.depth_camera_info,
        config.topics.semantic_points,
        config.topics.semantic_objects,
        config.topics.map_points,
        config.topics.diagnostics,
        config.topics.vo_odom,
    ]
    source = (config.pose.source or "").lower()
    if source in {"odom", "odometry"}:
        topics.append(config.topics.odom)
    elif source in {"pose", "pose_stamped", "posestamped"}:
        topics.append(config.topics.pose)
    if config.logging.rosbag_topics:
        topics.extend(config.logging.rosbag_topics)

    deduped = []
    seen = set()
    for topic in topics:
        if topic and topic not in seen:
            seen.add(topic)
            deduped.append(topic)
    return deduped


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
