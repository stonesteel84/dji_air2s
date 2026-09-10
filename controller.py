from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from .types import ControlDecision, Detection, RCCommand


@dataclass(frozen=True)
class TargetFollowerConfig:
    target_labels: Tuple[str, ...] = ('person', 'vehicle', 'car')
    min_confidence: float = 0.35
    desired_area_ratio: float = 0.08
    center_deadband: float = 0.06
    area_deadband: float = 0.16
    yaw_gain: float = 42.0
    vertical_gain: float = 36.0
    forward_gain: float = 24.0
    max_rc: int = 25
    command_smoothing: float = 0.55
    lost_hover_seconds: float = 0.8
    max_search_seconds: float = 12.0
    search_yaw: int = 12


class TargetFollower:
    """Image-plane target follower producing bounded Tello RC commands.

    This controller follows one detected target. It is deliberately not an
    obstacle-avoidance or collision-avoidance controller.
    """

    def __init__(self, config: Optional[TargetFollowerConfig] = None):
        self.config = config or TargetFollowerConfig()
        self._started_at: Optional[float] = None
        self._last_seen_at: Optional[float] = None
        self._last_command = RCCommand()

    def reset(self) -> None:
        self._started_at = None
        self._last_seen_at = None
        self._last_command = RCCommand()

    def update(
        self,
        detections: Iterable[Detection],
        frame_shape: Sequence[int],
        timestamp: float,
    ) -> ControlDecision:
        if len(frame_shape) < 2:
            raise ValueError('frame_shape must contain height and width')
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        if frame_h <= 0 or frame_w <= 0:
            raise ValueError('frame dimensions must be positive')

        now = float(timestamp)
        if self._started_at is None:
            self._started_at = now

        target = self._select_target(detections)
        if target is None:
            return self._on_target_lost(now)

        self._last_seen_at = now
        cx, cy = target.center
        horizontal_error = (cx - frame_w * 0.5) / (frame_w * 0.5)
        vertical_error = (cy - frame_h * 0.5) / (frame_h * 0.5)
        area_ratio = target.area / float(frame_w * frame_h)
        desired_area = max(self.config.desired_area_ratio, 1e-6)
        area_error = (desired_area - area_ratio) / desired_area

        yaw = self._axis_command(horizontal_error, self.config.center_deadband, self.config.yaw_gain)
        up_down = -self._axis_command(vertical_error, self.config.center_deadband, self.config.vertical_gain)
        forward = self._axis_command(area_error, self.config.area_deadband, self.config.forward_gain)
        raw = RCCommand(0, forward, up_down, yaw).clamped(self.config.max_rc)
        command = self._smooth(raw)
        return ControlDecision(
            command=command,
            mode='track',
            target=target,
            horizontal_error=horizontal_error,
            vertical_error=vertical_error,
            area_error=area_error,
            reason='target locked',
        )

    def _select_target(self, detections: Iterable[Detection]) -> Optional[Detection]:
        priorities = {label.lower(): i for i, label in enumerate(self.config.target_labels)}
        candidates = [
            det for det in detections
            if det.confidence >= self.config.min_confidence and det.label.lower() in priorities
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda det: (-priorities[det.label.lower()], det.confidence, det.area),
        )

    def _on_target_lost(self, now: float) -> ControlDecision:
        reference = self._last_seen_at if self._last_seen_at is not None else self._started_at
        elapsed = max(0.0, now - float(reference))
        self._last_command = RCCommand()
        if self._last_seen_at is not None and elapsed < self.config.lost_hover_seconds:
            return ControlDecision(RCCommand(), 'hover', reason='temporary target loss')
        if elapsed < self.config.max_search_seconds:
            search = RCCommand(yaw=self.config.search_yaw).clamped(self.config.max_rc)
            self._last_command = search
            return ControlDecision(search, 'search', reason='searching for configured target')
        return ControlDecision(
            RCCommand(),
            'land',
            request_land=True,
            reason='target search timeout',
        )

    @staticmethod
    def _axis_command(error: float, deadband: float, gain: float) -> int:
        if abs(error) <= max(0.0, deadband):
            return 0
        return int(round(float(gain) * float(error)))

    def _smooth(self, raw: RCCommand) -> RCCommand:
        alpha = max(0.0, min(1.0, float(self.config.command_smoothing)))
        previous = self._last_command

        def blend(old: int, new: int) -> int:
            if new == 0:
                return 0
            return int(round((1.0 - alpha) * old + alpha * new))

        command = RCCommand(
            blend(previous.left_right, raw.left_right),
            blend(previous.forward_backward, raw.forward_backward),
            blend(previous.up_down, raw.up_down),
            blend(previous.yaw, raw.yaw),
        ).clamped(self.config.max_rc)
        self._last_command = command
        return command
