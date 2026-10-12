# scripts/

Stage 3 tooling lands here, only if the 3.1 evaluation says pretrained pose is not good enough:

- `eval_pose.py` (3.1): wrist visibility fraction, confidence histograms, cycle recall vs a manual count, flicker score. Writes `docs/eval_v1.md`.
- `prelabel.py` (3.2): stratified 1,000-frame sample, run the current model, write CVAT XML with the 13-point skeleton.
- `train.sh` (3.4): CVAT XML to COCO, MMPose fine-tune of RTMPose-l with the 13-point head, export ONNX.
