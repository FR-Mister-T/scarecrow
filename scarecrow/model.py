import hashlib
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export.passes import move_to_device_pass

BUNDLED_WEIGHTS_FILENAME = "license-plate-finetune-v1n.pt2"
BUNDLED_WEIGHTS_SHA256 = "1404fd70f09f2c9fe20c292534b1821b7c8749421fae9cf9fd45a0279c4d9ce8"


def _verify_bundled_weights(path: str) -> None:
    """Raise if the file's SHA-256 does not match the pinned bundled weights hash."""
    with open(path, "rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != BUNDLED_WEIGHTS_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {BUNDLED_WEIGHTS_SHA256}, got {actual}. "
            f"Rename the file if it is not the bundled {BUNDLED_WEIGHTS_FILENAME}."
        )


def load(weights: str, device: str | None = None) -> nn.Module:
    """Load detection model with frozen weights."""
    if Path(weights).name == BUNDLED_WEIGHTS_FILENAME:
        _verify_bundled_weights(weights)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("CUDA not available, running on CPU (slower)")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not writable.*")
        ep = torch.export.load(weights)
    ep = move_to_device_pass(ep, device)
    model = ep.module()
    model.requires_grad_(False)
    return model


def detect(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable forward pass through the detector. images is a (B, 3, H, W)
    batch of RGB tensors in [0, 1].

    Returns boxes (B, N, 4) as (cx, cy, w, h) and each box's max class score (B, N).
    """
    raw = model(images)[0]
    # 4 box coords + nc class scores per prediction:
    # (B, 4+nc, N) -> (B, N, 4+nc)
    preds = raw.permute(0, 2, 1)
    return preds[..., :4], preds[..., 4:].max(dim=-1).values


def letterbox(images: torch.Tensor, imgsz: int) -> torch.Tensor:
    """Pad to square imgsz preserving aspect ratio."""
    b, _, h, w = images.shape
    if h == imgsz and w == imgsz:
        return images
    scale = imgsz / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = F.interpolate(images, size=(nh, nw), mode="bilinear", align_corners=False)
    # 114 is YOLO's standard letterbox padding value
    padded = torch.full((b, 3, imgsz, imgsz), 114 / 255, device=images.device)
    py, px = (imgsz - nh) // 2, (imgsz - nw) // 2
    padded[:, :, py : py + nh, px : px + nw] = resized
    return padded


def predict(
    model: nn.Module,
    img: np.ndarray,
    imgsz: int = 640,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.7,
) -> tuple[list[tuple[int, int, int, int]], float]:
    """Run detection with NMS. img: RGB uint8."""
    h, w = img.shape[:2]
    device = next(model.parameters()).device
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    padded = letterbox(tensor, imgsz)

    with torch.no_grad():
        boxes, scores = detect(model, padded)

    boxes, scores = boxes[0], scores[0]
    mask = scores > conf_thresh
    boxes, scores = boxes[mask], scores[mask]
    if len(scores) == 0:
        return [], 0.0

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    cx, cy, bw, bh = boxes.unbind(-1)
    xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)

    keep = _nms(xyxy, scores, iou_thresh)
    xyxy, scores = xyxy[keep], scores[keep]

    # undo letterboxing by mapping boxes back to original image coordinates
    scale = imgsz / max(h, w)
    pad_x = (imgsz - int(w * scale)) // 2
    pad_y = (imgsz - int(h * scale)) // 2
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale

    max_conf = float(scores.max())
    bboxes = [
        (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        for x1, y1, x2, y2 in xyxy.tolist()
    ]
    return bboxes, max_conf


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Drop boxes that overlap a higher scoring one by more than iou_thresh.
    Boxes are (N, 4) corners (x1, y1, x2, y2).
    """
    order = scores.argsort(descending=True)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = torch.max(boxes[i, 0], boxes[rest, 0])
        yy1 = torch.max(boxes[i, 1], boxes[rest, 1])
        xx2 = torch.min(boxes[i, 2], boxes[rest, 2])
        yy2 = torch.min(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thresh]
    return torch.stack(keep)
