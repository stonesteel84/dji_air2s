from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Detection:
    """Detector output expressed in original-frame pixel coordinates."""

    class_id: int
    label: str
    confidence: float
    xyxy: Tuple[float, float, float, float]

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class RCCommand:
    """Tello SDK `rc a b c d` command in the official channel order."""

    left_right: int = 0
    forward_backward: int = 0
    up_down: int = 0
    yaw: int = 0

    def clamped(self, limit: int = 100) -> 'RCCommand':
        limit = max(0, min(100, int(limit)))

        def clamp(value: int) -> int:
            return max(-limit, min(limit, int(round(value))))

        return RCCommand(
            clamp(self.left_right),
            clamp(self.forward_backward),
            clamp(self.up_down),
            clamp(self.yaw),
        )

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.left_right, self.forward_backward, self.up_down, self.yaw


@dataclass(frozen=True)
class ControlDecision:
    command: RCCommand
    mode: str
    target: Optional[Detection] = None
    horizontal_error: float = 0.0
    vertical_error: float = 0.0
    area_error: float = 0.0
    request_land: bool = False
    reason: str = ''
