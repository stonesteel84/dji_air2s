import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2

from .controller import TargetFollower, TargetFollowerConfig
from .detectors import RepositoryYoloDetector
from .simulation import annotate_frame
from .types import RCCommand


ARM_TOKEN = 'TELLO_EDU'


@dataclass(frozen=True)
class RealFlightConfig:
    weights: str
    data: str
    target_labels: Sequence[str]
    duration_seconds: float = 60.0
    imgsz: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    device: str = '0'
    half: bool = False
    min_takeoff_battery: int = 50
    min_control_height_cm: int = 40
    max_control_height_cm: int = 180
    command_hz: float = 10.0
    view: bool = True
    output: Optional[str] = None


def validate_arm_token(token: Optional[str]) -> None:
    if token != ARM_TOKEN:
        raise ValueError(
            f'Real takeoff is locked. Pass --arm-token {ARM_TOKEN} only after the propeller-off checks.'
        )


def apply_altitude_guard(
    command: RCCommand,
    height_cm: Optional[int],
    min_height_cm: int,
    max_height_cm: int,
) -> RCCommand:
    """Block unsafe vertical RC while preserving horizontal/yaw control."""

    if min_height_cm >= max_height_cm:
        raise ValueError('min_height_cm must be lower than max_height_cm')
    up_down = command.up_down
    if height_cm is None:
        up_down = 0
    elif height_cm <= min_height_cm and up_down < 0:
        up_down = 0
    elif height_cm >= max_height_cm and up_down > 0:
        up_down = 0
    return RCCommand(
        command.left_right,
        command.forward_backward,
        up_down,
        command.yaw,
    )


def run_real(config: RealFlightConfig, allow_takeoff: bool = False, arm_token: Optional[str] = None) -> dict:
    """Run observation or target following against a real Tello EDU.

    With allow_takeoff=False this opens video and performs detection only. A real
    takeoff requires both allow_takeoff=True and the explicit arm token.
    """

    if allow_takeoff:
        validate_arm_token(arm_token)
    if config.min_control_height_cm >= config.max_control_height_cm:
        raise ValueError('min_control_height_cm must be lower than max_control_height_cm')

    from djitellopy import Tello, TelloException

    tello = Tello()
    detector = RepositoryYoloDetector(
        weights=config.weights,
        data=config.data,
        target_labels=config.target_labels,
        imgsz=config.imgsz,
        confidence=config.confidence,
        iou=config.iou,
        device=config.device,
        half=config.half,
    )
    controller = TargetFollower(
        TargetFollowerConfig(
            target_labels=tuple(config.target_labels),
            min_confidence=config.confidence,
        )
    )
    frame_reader = None
    video_writer = None
    airborne = False
    frames = 0
    detections_total = 0
    last_height_cm = None
    started = None
    last_command_at = 0.0
    last_frame = None

    try:
        detector.load()
        tello.connect()
        battery = int(tello.get_battery())
        if allow_takeoff and battery < config.min_takeoff_battery:
            raise RuntimeError(
                f'Battery {battery}% is below the takeoff threshold {config.min_takeoff_battery}%.'
            )
        tello.streamon()
        frame_reader = tello.get_frame_read()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            last_frame = frame_reader.frame
            if last_frame is not None and getattr(last_frame, 'size', 0) > 0:
                break
            time.sleep(0.05)
        if last_frame is None or getattr(last_frame, 'size', 0) == 0:
            raise RuntimeError('No Tello video frame arrived within 10 seconds.')

        if config.output:
            output = Path(config.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            h, w = last_frame.shape[:2]
            video_writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*'mp4v'), config.command_hz, (w, h)
            )
            if not video_writer.isOpened():
                raise RuntimeError(f'Could not open output video: {output}')

        if allow_takeoff:
            tello.takeoff()
            airborne = True

        started = time.monotonic()
        while time.monotonic() - started < config.duration_seconds:
            frame = frame_reader.frame
            if frame is None or getattr(frame, 'size', 0) == 0:
                time.sleep(0.01)
                continue
            now = time.monotonic()
            detections = detector.detect(frame)
            decision = controller.update(detections, frame.shape, now)
            frames += 1
            detections_total += len(detections)

            if airborne and now - last_command_at >= 1.0 / max(config.command_hz, 1.0):
                try:
                    last_height_cm = int(tello.get_height())
                except (KeyError, TypeError, ValueError, TelloException):
                    last_height_cm = None
                guarded_command = apply_altitude_guard(
                    decision.command,
                    last_height_cm,
                    config.min_control_height_cm,
                    config.max_control_height_cm,
                )
                tello.send_rc_control(*guarded_command.as_tuple())
                last_command_at = now

            annotated = annotate_frame(frame, decision)
            if video_writer is not None:
                video_writer.write(annotated)
            if config.view:
                cv2.imshow('Tello EDU object detection', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('l')):
                    break
                if key == ord('e') and airborne:
                    tello.emergency()
                    airborne = False
                    break
            if decision.request_land and airborne:
                break

        return {
            'mode': 'flight' if allow_takeoff else 'observe',
            'frames': frames,
            'detections': detections_total,
            'battery_at_start': battery,
            'last_height_cm': last_height_cm,
            'duration_seconds': round(time.monotonic() - started, 2),
        }
    finally:
        if video_writer is not None:
            video_writer.release()
        if config.view:
            cv2.destroyAllWindows()
        if airborne:
            try:
                tello.send_rc_control(*RCCommand().as_tuple())
                tello.land()
            finally:
                airborne = False
        try:
            if getattr(tello, 'stream_on', False):
                tello.streamoff()
        finally:
            tello.end()
