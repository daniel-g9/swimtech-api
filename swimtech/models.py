"""Model loading and export for the detector (1.3) and, later, the pose model (2.1).

Everything that imports torch or rfdetr lives behind a function so that `swimtech ingest`
and the tests run on a laptop without the `models` extra installed.

Install the heavy dependencies with:  uv sync --extra models

Teaching notes
--------------
RF-DETR ships several sizes. Bigger is more accurate on hard cases (half-submerged
bodies, splash) but slower; on a laptop CPU "nano" runs a 4K clip sampled at 15 fps in a
few minutes, "medium" in tens of minutes. The GPU worker (5.2) uses "medium". Every
variant is COCO-pretrained, where class id 1 is `person` (COCO ids are 1-based with
gaps, which is why we never assume 0 = person).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("swimtech.models")

MODEL_DIR = Path(os.environ.get("SWIMTECH_MODEL_DIR", "models"))

# Our short names -> rfdetr class names. The number is the square input resolution the
# variant was trained at (from the rfdetr model table); kept for reference and export.
RFDETR_VARIANTS: dict[str, tuple[str, int]] = {
    "rfdetr-nano": ("RFDETRNano", 384),
    "rfdetr-small": ("RFDETRSmall", 512),
    "rfdetr-medium": ("RFDETRMedium", 576),
    "rfdetr-large": ("RFDETRLarge", 704),
}
COCO_PERSON_CLASS_ID = 1


class ModelsNotInstalledError(RuntimeError):
    pass


def _import_rfdetr() -> Any:
    try:
        import rfdetr
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ModelsNotInstalledError(
            "rfdetr is not installed. Run `uv sync --extra models` (several GB; torch)."
        ) from e
    return rfdetr


def load_rfdetr(name: str = "rfdetr-medium", device: str | None = None) -> Any:
    """Instantiate a COCO-pretrained RF-DETR. Downloads weights on first use."""
    if name not in RFDETR_VARIANTS:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(RFDETR_VARIANTS)}")
    rfdetr = _import_rfdetr()
    cls_name, _ = RFDETR_VARIANTS[name]
    cls = getattr(rfdetr, cls_name)
    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    log.info("loading %s (%s)", name, cls_name)
    model = cls(**kwargs)
    try:
        # Fuses layers / compiles for a faster predict(); harmless if unsupported.
        model.optimize_for_inference()
    except Exception as e:  # noqa: BLE001 - optimization is best-effort
        log.debug("optimize_for_inference skipped: %s", e)
    return model


def export_rfdetr_onnx(name: str = "rfdetr-medium", out_dir: Path = MODEL_DIR) -> Path:
    """Export a detector to ONNX for the serving worker (5.2). Returns the file path.

    Serving from ONNX needs the same preprocessing rfdetr applies in predict() (resize
    to the square training resolution, ImageNet mean/std). That runtime path is written
    in 5.2 against a real exported file; until then torch inference is used everywhere.
    """
    model = load_rfdetr(name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = model.export(output_dir=str(out_dir), format="onnx", verbose=False)
    log.info("exported %s -> %s", name, path)
    return Path(path)
