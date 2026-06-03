#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from scarecrow import frame, ocr
from scarecrow import model as yolo
from scarecrow.io import image_paths, load, load_pattern, save, save_pattern
from scarecrow.model import DEFAULT_WEIGHTS_FILENAME
from scarecrow.optimize import MIN_PLATE_WIDTH, Config, optimize


def _default_pattern_path(input_path: str) -> str:
    """Default generated pattern path for an input image."""
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}_pattern.png"))


def _cmd_generate(args) -> int:
    config = Config(steps=args.steps, seed=args.seed)

    def on_step(step, loss):
        if step % 25 == 0 or step == config.steps - 1:
            print(f"step {step:4d}  det={loss:.4f}")

    pattern = optimize(args.input, args.weights, config, on_step=on_step)
    pattern = pattern.cpu().numpy()
    out = args.output or _default_pattern_path(args.input)
    save_pattern(pattern, out)
    print(f"Saved pattern to {out}")
    return 0


def _cmd_apply(args) -> int:
    img = load(args.input)
    pattern = load_pattern(args.pattern)

    model = yolo.load(args.weights)
    bboxes, _ = yolo.predict(model, img)
    if not bboxes:
        print("No plate detected", file=sys.stderr)
        return 1

    frame.apply_pattern(img, pattern, bboxes)
    out = args.output or str(Path(args.input).with_stem(Path(args.input).stem + "_framed"))
    save(img, out)
    print(f"Saved to {out}")
    return 0


def _cmd_eval(args) -> int:
    yolo_model = yolo.load(args.weights)
    pattern = load_pattern(args.pattern)
    paths = image_paths(Path(args.input))
    if not paths:
        print(f"No images in {args.input}", file=sys.stderr)
        return 1

    ocr_reader = None
    if args.ocr:
        try:
            ocr_reader = ocr.load_reader()
        except ImportError:
            print("rapidocr-onnxruntime required for --ocr. Install: uv sync --extra ocr", file=sys.stderr)
            return 1

    total, evaded, tiny_skipped = 0, 0, 0
    conf_clean_sum, conf_adv_sum = 0.0, 0.0
    ocr_total, ocr_changed = 0, 0
    rows = []

    for p in paths:
        img = load(p)
        bboxes, clean_conf = yolo.predict(yolo_model, img)
        n_before = len(bboxes)
        bboxes = [b for b in bboxes if b[2] >= MIN_PLATE_WIDTH]
        skipped_tiny = n_before - len(bboxes)
        tiny_skipped += skipped_tiny
        if not bboxes:
            row = {
                "path": str(p),
                "clean_boxes": 0,
                "attacked_boxes": 0,
                "clean_conf": clean_conf,
                "attacked_conf": 0.0,
                "evaded": False,
                "skipped_tiny": skipped_tiny,
                "ocr": [],
            }
            rows.append(row)
            if not args.json:
                print(f"{p.name}  [no detection on clean]")
            continue

        total += 1
        adv = img.copy()
        frame.apply_pattern(adv, pattern, bboxes)
        adv_bboxes, adv_conf = yolo.predict(yolo_model, adv)
        adv_bboxes = [b for b in adv_bboxes if b[2] >= MIN_PLATE_WIDTH]
        was_evaded = len(adv_bboxes) == 0
        ocr_rows = []

        if was_evaded:
            evaded += 1
        conf_clean_sum += clean_conf
        conf_adv_sum += adv_conf

        if ocr_reader is not None:
            for bbox in bboxes:
                clean_crop = ocr.crop_for_ocr(img, bbox)
                adv_crop = ocr.crop_for_ocr(adv, bbox)
                clean_text = ocr.read_plate(ocr_reader, clean_crop)
                adv_text = ocr.read_plate(ocr_reader, adv_crop)
                if len(clean_text) >= 2:
                    ocr_total += 1
                    changed = clean_text != adv_text
                    if changed:
                        ocr_changed += 1
                    ocr_rows.append({"clean": clean_text, "framed": adv_text, "changed": changed})

        rows.append(
            {
                "path": str(p),
                "clean_boxes": len(bboxes),
                "attacked_boxes": len(adv_bboxes),
                "clean_conf": clean_conf,
                "attacked_conf": adv_conf,
                "evaded": was_evaded,
                "skipped_tiny": skipped_tiny,
                "ocr": ocr_rows,
            }
        )

        if not args.json:
            status = "EVADED" if was_evaded else f"conf {clean_conf:.3f} -> {adv_conf:.3f}"
            if ocr_rows:
                parts = [
                    f'"{o["clean"]}" -> "{o["framed"]}"' if o["changed"]
                    else f'"{o["clean"]}" [unchanged]'
                    for o in ocr_rows
                ]
                status += "  OCR: " + ", ".join(parts)
            print(f"{p.name}  {status}")

    if args.json:
        summary = {
            "total": total,
            "evaded": evaded,
            "mean_clean_conf": conf_clean_sum / total if total else 0.0,
            "mean_attacked_conf": conf_adv_sum / total if total else 0.0,
            "tiny_skipped": tiny_skipped,
            "ocr_total": ocr_total,
            "ocr_changed": ocr_changed,
        }
        print(
            json.dumps(
                {
                    "input": str(args.input),
                    "pattern": str(args.pattern),
                    "weights": str(args.weights),
                    "ocr": args.ocr,
                    "images": rows,
                    "summary": summary,
                },
                indent=2,
            )
        )
        return 0

    print("---")
    if total > 0:
        print(
            f"Evasion: {evaded}/{total} ({100 * evaded / total:.0f}%)"
            f" | Mean conf: {conf_clean_sum / total:.3f} -> {conf_adv_sum / total:.3f}"
        )
    if ocr_reader is not None and ocr_total > 0:
        print(f"OCR changed: {ocr_changed}/{ocr_total} ({100 * ocr_changed / ocr_total:.0f}%)")
    if tiny_skipped:
        print(f"(skipped {tiny_skipped} detections < {MIN_PLATE_WIDTH}px)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Adversarial license plate frame generator.")
    sub = p.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate adversarial frame pattern")
    gen.add_argument("input", metavar="IMAGE", help="Plate image file")
    gen.add_argument("--weights", metavar="MODEL.pt2", default=DEFAULT_WEIGHTS_FILENAME, help="Detector .pt2 file")
    gen.add_argument("--steps", metavar="N", type=int, default=1000, help="Optimization steps")
    gen.add_argument("--seed", metavar="N", type=int, default=None, help="Reproducible optimization seed")
    gen.add_argument("-o", "--output", metavar="PATTERN.png", help="Output pattern path")

    ap = sub.add_parser("apply", help="Apply pattern to a plate image")
    ap.add_argument("input", metavar="IMAGE", help="Input image")
    ap.add_argument("--pattern", metavar="PATTERN.png", required=True, help="Generated frame pattern PNG")
    ap.add_argument("--weights", metavar="MODEL.pt2", default=DEFAULT_WEIGHTS_FILENAME, help="Detector .pt2 file")
    ap.add_argument("-o", "--output", metavar="OUT", help="Output image path")

    ev = sub.add_parser("eval", help="Evaluate pattern effectiveness")
    ev.add_argument("input", metavar="INPUT", help="Image file or directory")
    ev.add_argument("--pattern", metavar="PATTERN.png", required=True, help="Generated frame pattern PNG")
    ev.add_argument("--weights", metavar="MODEL.pt2", default=DEFAULT_WEIGHTS_FILENAME, help="Detector .pt2 file")
    ev.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    ev.add_argument("--ocr", action="store_true", help="Also run RapidOCR on clean/framed plate crops")

    args = p.parse_args()
    cmd = {"generate": _cmd_generate, "apply": _cmd_apply, "eval": _cmd_eval}
    return cmd[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
