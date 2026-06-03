<img align="left" width="120" src="assets/scarecrow.png" alt="scarecrow">

# scarecrow

Adversarial frame pattern optimization for evading ALPR (automated license plate recognition). Given a photo of your plate, scarecrow generates a printable grayscale pattern that suppresses detection while keeping the plate readable to humans. [Keeps the flock away.](https://www.eff.org/deeplinks/2025/12/effs-investigations-expose-flock-safetys-surveillance-abuses-2025-review)

> [!WARNING]
> This project is a research tool for personal privacy against warrantless mass surveillance. It is not intended for evading law enforcement in the commission of a crime. Frame patterns do not obstruct or alter plate text, but laws around devices placed near the plate vary by jurisdiction and are evolving. Please check your local laws before use.

## Why

Flock Safety and other ALPR cameras are in thousands of our neighborhoods, parking lots, and police networks across the US. They capture and index every plate that passes, feeding a searchable surveillance database with no warrant, no notification, and in most cases no public oversight.

A system that can track anyone, anywhere, with no transparency or accountability is fundamentally immoral. This project is my way of exploring what can be done about it, ethically and legally.

Inspired by Ben Jordan's [PlateShapez](https://github.com/bennjordan/PlateShapez) and his investigations into Flock Safety. Where his approach uses random geometric perturbations on the plate, scarecrow uses gradient-based optimization of a frame pattern around it, aiming to be more robust and legally viable since the plate itself is never altered.

## Results

On the included test plate, scarecrow drops detection confidence from 0.84 to 0.00 (full evasion) in 1000 steps, and the plate remains human-readable.

| Before | After |
|---|---|
| ![before](assets/before.jpg) | ![after](assets/after.jpg) |

> [!NOTE]
> OCR is sometimes corrupted as a side effect, roughly 40% of the time depending on the random seed.

## How It Works

Scarecrow optimizes a grayscale frame pattern using gradient descent against a YOLO plate detection model. The pattern sits in the border region around the plate, inside a printable frame, and is tuned specifically to minimize the detector's confidence on your specific plate.

To keep the pattern from overfitting to the reference photo, each optimization step applies random augmentations that simulate what a camera might actually see:

- **Radial lens distortion:** barrel/pincushion from real camera optics
- **Rotation & perspective warp:** different viewing angles
- **Brightness & contrast shifts:** varying lighting and IR illumination
- **Gaussian blur:** camera motion and focus
- **Additive noise:** sensor noise in low light
- **Scale jitter:** different distances from the camera

Flock and most ALPR cameras are rear-facing and [mounted at 8 to 12 feet](https://www.flocksafety.com/implementation-guide), so the viewing geometry is fairly constrained. The augmentation ranges were chosen with this in mind: rotation stays within 10 degrees, perspective within 20 to 25 degrees, and scale varies from 0.5x to 1.2x to cover plates captured at different distances from the camera.

Optimizing the pattern across this whole range of transformations is called Expectation over Transformation (EoT), and the loss uses logsumexp to upweight the hardest samples, so optimization focuses on the conditions where the pattern is weakest.

The included detection model is a [YOLO11n](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) plate detector exported via `torch.export`. If you're targeting a different detector, see [Using your own detection model](#using-your-own-detection-model) below.

## Usage

Requires Python 3.11+. A CUDA GPU is recommended but not required, as optimization also works on CPU (slower).

Install dependencies:

```bash
uv sync
```

Take a photo of your plate from the front, straight on, with even lighting and minimal angle. This is the reference image the optimization works from. See `test_plate.jpg` for an example.

```bash
# Generate a frame pattern for your plate (takes a few minutes on GPU)
scarecrow generate plate.jpg

# Reproducible generation with a fixed seed
scarecrow generate plate.jpg --seed 42

# Preview the result
scarecrow apply plate.jpg --pattern plate_pattern.png

# Evaluate detection evasion
scarecrow eval plate.jpg --pattern plate_pattern.png

# Evaluate RapidOCR reads (requires rapidocr: uv sync --extra ocr)
scarecrow eval plate.jpg --pattern plate_pattern.png --ocr

# Emit structured eval results
scarecrow eval plate.jpg --pattern plate_pattern.png --json
```

## Using your own detection model

> [!WARNING]
> `torch.export.load` uses pickle, so loading an untrusted `.pt2` can execute arbitrary code. Only use `--weights` from sources you trust.

<details>
<summary>Convert ultralytics weights and point scarecrow at them</summary>

Scarecrow works with any plate detection model, not just the included YOLO11n. The model needs to be in `torch.export` format (`.pt2`).

If you have ultralytics `.pt` weights, you can convert them like this:

```bash
uv run --with ultralytics python3 -c "
import torch; from ultralytics import YOLO
m = YOLO('your-model.pt').model.eval()
for p in m.parameters(): p.requires_grad_(False)
ep = torch.export.export(m, (torch.randn(1, 3, 640, 640),))
torch.export.save(ep, 'your-model.pt2')
"
```

Then pass `--weights your-model.pt2` to any scarecrow command.

</details>

## Limitations

- I haven't tested this against a real ALPR camera, only in simulation against rendered composites. If you have access to Flock or other ALPR hardware and can benchmark, I'd love to hear how it performs.
- The included model is a single YOLO11n plate detector, and adversarial patterns can transfer across similar architectures, but how well they transfer to other detectors (including Flock Safety's proprietary YOLO variant) is untested.

## License

GPL-3.0. See [LICENSE](LICENSE).
