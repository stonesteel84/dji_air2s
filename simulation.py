import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .controller import TargetFollower, TargetFollowerConfig
from .detectors import ColorTargetDetector
from .types import ControlDecision, RCCommand


@dataclass(frozen=True)
class SimulationConfig:
    width: int = 640
    height: int = 480
    fps: int = 20
    duration_seconds: float = 14.0
    seed: int = 7
    initial_center_x: float = 0.82
    initial_center_y: float = 0.24
    initial_area_ratio: float = 0.018
    target_aspect_ratio: float = 0.75
    yaw_response: float = 0.75
    vertical_response: float = 0.65
    approach_response: float = 1.55
    disturbance: float = 0.004


class ImagePlaneSimulator:
    """Deterministic camera-target simulator for controller integration tests."""

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.center_x = float(self.config.initial_center_x)
        self.center_y = float(self.config.initial_center_y)
        self.area_ratio = float(self.config.initial_area_ratio)
        self.elapsed = 0.0
        self._rng = np.random.default_rng(self.config.seed)

    def render(self) -> np.ndarray:
        width, height = self.config.width, self.config.height
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (32, 38, 45)
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (45, 52, 61), 1)
        for y in range(0, height, 60):
            cv2.line(frame, (0, y), (width, y), (45, 52, 61), 1)

        aspect = max(0.1, self.config.target_aspect_ratio)
        box_h = math.sqrt(max(self.area_ratio, 1e-6) * width * height / aspect)
        box_w = aspect * box_h
        cx, cy = self.center_x * width, self.center_y * height
        x1 = int(round(cx - box_w * 0.5))
        y1 = int(round(cy - box_h * 0.5))
        x2 = int(round(cx + box_w * 0.5))
        y2 = int(round(cy + box_h * 0.5))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 3)
        cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 255), -1)
        cv2.putText(frame, 'SIM TARGET', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 220, 255), 2, cv2.LINE_AA)
        return frame

    def step(self, command: RCCommand, dt: float) -> None:
        dt = max(0.0, float(dt))
        yaw = command.yaw / 100.0
        vertical = command.up_down / 100.0
        forward = command.forward_backward / 100.0
        self.center_x -= self.config.yaw_response * yaw * dt
        self.center_y += self.config.vertical_response * vertical * dt
        self.area_ratio *= math.exp(self.config.approach_response * forward * dt)

        phase = self.elapsed * 1.7
        self.center_x += self.config.disturbance * math.sin(phase) * dt
        self.center_y += self.config.disturbance * math.cos(phase * 0.8) * dt
        self.center_x += float(self._rng.normal(0.0, self.config.disturbance * 0.03))
        self.center_y += float(self._rng.normal(0.0, self.config.disturbance * 0.03))
        self.center_x = float(np.clip(self.center_x, 0.05, 0.95))
        self.center_y = float(np.clip(self.center_y, 0.05, 0.95))
        self.area_ratio = float(np.clip(self.area_ratio, 0.002, 0.35))
        self.elapsed += dt


def annotate_frame(frame: np.ndarray, decision: ControlDecision) -> np.ndarray:
    output = frame.copy()
    h, w = output.shape[:2]
    cv2.drawMarker(output, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 26, 2)
    if decision.target is not None:
        x1, y1, x2, y2 = (int(round(x)) for x in decision.target.xyxy)
        cv2.rectangle(output, (x1, y1), (x2, y2), (60, 255, 60), 2)
    command = decision.command
    lines = [
        f'mode={decision.mode}',
        f'rc lr/fb/ud/yaw={command.left_right}/{command.forward_backward}/{command.up_down}/{command.yaw}',
        f'error x/y/area={decision.horizontal_error:+.3f}/{decision.vertical_error:+.3f}/{decision.area_error:+.3f}',
    ]
    for index, text in enumerate(lines):
        cv2.putText(output, text, (12, 26 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def run_simulation(
    output_dir: Path,
    config: Optional[SimulationConfig] = None,
    controller_config: Optional[TargetFollowerConfig] = None,
    save_video: bool = True,
    view: bool = False,
) -> Dict[str, object]:
    config = config or SimulationConfig()
    controller_config = controller_config or TargetFollowerConfig(target_labels=('target',))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simulator = ImagePlaneSimulator(config)
    detector = ColorTargetDetector()
    controller = TargetFollower(controller_config)
    dt = 1.0 / max(1, config.fps)
    total_steps = max(1, int(round(config.duration_seconds * config.fps)))
    trace = []
    video_path = output_dir / 'tello_closed_loop_sim.mp4'
    writer = None
    if save_video:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            float(config.fps),
            (config.width, config.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f'Could not open simulation video writer: {video_path}')

    try:
        for step in range(total_steps):
            timestamp = step * dt
            frame = simulator.render()
            detections = detector.detect(frame)
            decision = controller.update(detections, frame.shape, timestamp)
            annotated = annotate_frame(frame, decision)
            if writer is not None:
                writer.write(annotated)
            if view:
                cv2.imshow('Tello EDU closed-loop simulation', annotated)
                if cv2.waitKey(max(1, int(round(1000 / config.fps)))) & 0xFF == ord('q'):
                    break
            trace.append({
                'step': step,
                'time': round(timestamp, 4),
                'mode': decision.mode,
                'center_x': simulator.center_x,
                'center_y': simulator.center_y,
                'area_ratio': simulator.area_ratio,
                'horizontal_error': decision.horizontal_error,
                'vertical_error': decision.vertical_error,
                'area_error': decision.area_error,
                'forward_backward': decision.command.forward_backward,
                'up_down': decision.command.up_down,
                'yaw': decision.command.yaw,
            })
            simulator.step(decision.command, dt)
            if decision.request_land:
                break
    finally:
        if writer is not None:
            writer.release()
        if view:
            cv2.destroyAllWindows()

    last = trace[-1]
    center_error = math.hypot(float(last['horizontal_error']), float(last['vertical_error']))
    area_error = abs(float(last['area_error']))
    success = center_error <= 0.10 and area_error <= 0.24
    summary = {
        'success': success,
        'steps': len(trace),
        'duration_seconds': round(len(trace) * dt, 3),
        'final_center_error': round(center_error, 5),
        'final_area_error': round(area_error, 5),
        'final_target_state': {
            'center_x': round(simulator.center_x, 5),
            'center_y': round(simulator.center_y, 5),
            'area_ratio': round(simulator.area_ratio, 5),
        },
        'controller': asdict(controller_config),
        'simulation': asdict(config),
        'video': str(video_path.resolve()) if save_video else None,
    }
    (output_dir / 'simulation_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    with (output_dir / 'simulation_trace.csv').open('w', newline='', encoding='utf-8') as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=list(trace[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(trace)
    return summary
