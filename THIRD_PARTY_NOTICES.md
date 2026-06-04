# Third-Party Notices

## Bundled Detector Model

Scarecrow bundles a converted detector model for the default `--weights` behavior.

- Upstream model: `morsetechlab/yolov11-license-plate-detection`
- Upstream URL: <https://huggingface.co/morsetechlab/yolov11-license-plate-detection>
- Upstream revision: `251a30d7daedca065f56e04b0af04052c907c68f`
- Upstream artifact: `license-plate-finetune-v1n.pt`
- Local artifact: `scarecrow/data/license-plate-finetune-v1n.pt2`
- Local SHA-256: `1404fd70f09f2c9fe20c292534b1821b7c8749421fae9cf9fd45a0279c4d9ce8`
- License: `AGPL-3.0-only`

The local artifact is a `torch.export` conversion of the upstream weights.
The upstream model card declares license `agpl-3.0`, and this distribution
records that bundled model under the SPDX identifier `AGPL-3.0-only`.
