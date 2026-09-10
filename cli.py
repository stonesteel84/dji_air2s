import argparse
import json
from pathlib import Path

from .controller import TargetFollowerConfig
from .real_flight import ARM_TOKEN, RealFlightConfig, run_real
from .simulation import SimulationConfig, run_simulation


ROOT = Path(__file__).resolve().parents[1]
FFCA_ROOT = Path(__file__).resolve().parent / 'ffca_yolo'


def _target_labels(value: str):
    labels = tuple(part.strip() for part in value.split(',') if part.strip())
    if not labels:
        raise argparse.ArgumentTypeError('at least one target label is required')
    return labels


def _add_real_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--weights', default=str(FFCA_ROOT / 'weights/best.pt'))
    parser.add_argument('--data', default=str(FFCA_ROOT / 'data/AITOD.yaml'))
    parser.add_argument('--targets', type=_target_labels, default=('person', 'vehicle'))
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='0')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--min-battery', type=int, default=50)
    parser.add_argument('--min-height', type=int, default=40)
    parser.add_argument('--max-height', type=int, default=180)
    parser.add_argument('--no-view', action='store_true')
    parser.add_argument('--output')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Tello EDU autonomous object-detection program')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    sim = subparsers.add_parser('simulate', help='run deterministic closed-loop simulation')
    sim.add_argument('--output-dir', default=str(ROOT / 'outputs/tello_sim'))
    sim.add_argument('--duration', type=float, default=14.0)
    sim.add_argument('--fps', type=int, default=20)
    sim.add_argument('--seed', type=int, default=7)
    sim.add_argument('--view', action='store_true')
    sim.add_argument('--no-video', action='store_true')

    observe = subparsers.add_parser('observe', help='real Tello video and detection; no flight commands')
    _add_real_arguments(observe)

    fly = subparsers.add_parser('fly', help='armed real target-follow flight')
    _add_real_arguments(fly)
    fly.add_argument('--arm-token', required=True, help=f'must be exactly {ARM_TOKEN}')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.mode == 'simulate':
        summary = run_simulation(
            Path(args.output_dir),
            config=SimulationConfig(duration_seconds=args.duration, fps=args.fps, seed=args.seed),
            controller_config=TargetFollowerConfig(target_labels=('target',)),
            save_video=not args.no_video,
            view=args.view,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary['success'] else 2

    config = RealFlightConfig(
        weights=args.weights,
        data=args.data,
        target_labels=args.targets,
        duration_seconds=args.duration,
        imgsz=args.imgsz,
        confidence=args.conf_thres,
        iou=args.iou_thres,
        device=args.device,
        half=args.half,
        min_takeoff_battery=args.min_battery,
        min_control_height_cm=args.min_height,
        max_control_height_cm=args.max_height,
        view=not args.no_view,
        output=args.output,
    )
    result = run_real(
        config,
        allow_takeoff=args.mode == 'fly',
        arm_token=getattr(args, 'arm_token', None),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
