# Bundled FFCA-YOLO inference runtime

This directory contains only the model code, inference utilities, AI-TOD class configuration, and
the TS-RPST checkpoint required by `tello_autonomy.ffca_detector`.

- `models/`: inherited YOLOv5 and FFCA/TS-RPST layers
- `utils/`: inherited preprocessing, NMS, box scaling, and device helpers
- `data/AITOD.yaml`: inference class names
- `weights/best.pt`: default TS-RPST checkpoint

Training scripts and the AI-TOD dataset are intentionally not bundled. Retrain in the original
FFCA-YOLO repository, then replace `weights/best.pt` or pass another path with `--weights`.
