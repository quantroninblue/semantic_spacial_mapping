from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Optional

from sensor_msgs.msg import CameraInfo, Image

from .converters import stamp_to_sec


class ApproximateRgbdSync:
    def __init__(
        self,
        queue_size: int,
        slop_sec: float,
        max_fps: float,
        callback: Callable[[Image, Optional[Image], Optional[CameraInfo], Optional[CameraInfo]], None],
    ):
        self.queue_size = max(int(queue_size), 1)
        self.slop_sec = float(slop_sec)
        self.min_period = 1.0 / float(max_fps) if max_fps and max_fps > 0.0 else 0.0
        self.callback = callback
        self.rgb_queue: Deque[Image] = deque(maxlen=self.queue_size)
        self.depth_queue: Deque[Image] = deque(maxlen=self.queue_size)
        self.rgb_info: Optional[CameraInfo] = None
        self.depth_info: Optional[CameraInfo] = None
        self.last_emit_time = -1.0
        self.sync_failures = 0
        self.dropped_frames = 0

    def add_rgb(self, msg: Image) -> None:
        self.rgb_queue.append(msg)
        self._try_emit()

    def add_depth(self, msg: Image) -> None:
        self.depth_queue.append(msg)
        self._try_emit()

    def set_rgb_info(self, msg: CameraInfo) -> None:
        self.rgb_info = msg

    def set_depth_info(self, msg: CameraInfo) -> None:
        self.depth_info = msg

    def _try_emit(self) -> None:
        if not self.rgb_queue:
            return

        rgb_msg = self.rgb_queue[-1]
        rgb_time = stamp_to_sec(rgb_msg.header.stamp)
        if self.min_period > 0.0 and self.last_emit_time > 0.0:
            if rgb_time - self.last_emit_time < self.min_period:
                self.dropped_frames += 1
                self.rgb_queue.clear()
                return

        depth_msg = self._nearest_depth(rgb_time)
        if self.depth_queue and depth_msg is None:
            self.sync_failures += 1
            return

        self.rgb_queue.clear()
        if depth_msg is not None:
            self._drop_depth_until(depth_msg)

        self.last_emit_time = rgb_time
        self.callback(rgb_msg, depth_msg, self.rgb_info, self.depth_info)

    def _nearest_depth(self, rgb_time: float) -> Optional[Image]:
        if not self.depth_queue:
            return None
        nearest = min(
            self.depth_queue,
            key=lambda msg: abs(stamp_to_sec(msg.header.stamp) - rgb_time),
        )
        if abs(stamp_to_sec(nearest.header.stamp) - rgb_time) > self.slop_sec:
            return None
        return nearest

    def _drop_depth_until(self, depth_msg: Image) -> None:
        while self.depth_queue:
            current = self.depth_queue.popleft()
            if current is depth_msg:
                break
