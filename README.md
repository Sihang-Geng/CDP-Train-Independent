<div align="center">

# CDP-Train

**COCO-driven training and checkpoint selection for object detection experiments**

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white">
  <img alt="Task" src="https://img.shields.io/badge/task-object%20detection-2E7D32">
  <img alt="Metric" src="https://img.shields.io/badge/metric-COCO%20AP-orange">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-AGPL--3.0-blue"></a>
</p>

</div>

CDP-Train is a research codebase for object detection experiments where the training checkpoint is selected by the same COCO AP metric used for paper evaluation. The repository adds training-time COCO API evaluation, COCO-based `best.pt` selection, custom annotation matching, and several experiment utilities on top of a YOLO training pipeline.

The main motivation is simple: in detection experiments, comparing models only at a fixed final epoch can mix model quality with convergence speed. This code makes the training loop periodically run COCO evaluation and uses `mAP50-95(B)` to decide which checkpoint should be treated as the best model.

## What This Repository Contains

| Part | Purpose |
| --- | --- |
| COCO AP checkpointing | Use COCO `mAP50-95(B)` as the fitness value for `best.pt`. |
| Scheduled COCO evaluation | Run COCO API every `N` epochs instead of every epoch to control overhead. |
| Best-checkpoint guard | Prevent non-COCO validation epochs from overwriting a COCO-selected `best.pt`. |
| COCO annotation lookup | Search common custom COCO JSON locations automatically. |
| Image-id alignment | Map prediction `image_id` back to annotation JSON IDs for non-numeric filenames. |
| Optimizer fallback | Fall back to standard `torch.optim.SGD` when custom optimizer paths are unavailable. |
| Utility scripts | Training, COCO testing, qualitative visualization, and plotting scripts. |

## Paper Status

The corresponding paper is currently under peer review. The full architecture figure and more detailed methodological description will be released after acceptance. This repository currently focuses on the reproducible training/evaluation code and the core checkpoint-selection protocol.

## Repository Layout

```text
CDP-Train/
├─ ultralytics/
│  ├─ engine/
│  │  ├─ trainer.py          # COCO fitness and best.pt control
│  │  └─ validator.py        # scheduled COCO evaluation
│  ├─ models/yolo/detect/
│  │  └─ val.py              # COCO API evaluation and image-id mapping
│  ├─ cfg/default.yaml       # CDP-related training arguments
│  ├─ train.py               # example training entry
│  ├─ visual.py              # qualitative visualization
│  ├─ coco-test.py           # COCO evaluation check
│  └─ plotfig2.py            # plotting utility
├─ FAIR_COMPARISON_IMPLEMENTATION.md
├─ pyproject.toml
└─ LICENSE
```

For implementation-level notes, see [`FAIR_COMPARISON_IMPLEMENTATION.md`](FAIR_COMPARISON_IMPLEMENTATION.md).

## Environment

Create a clean environment first:

```bash
conda create -n cdp-train python=3.10 -y
conda activate cdp-train
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install this repository in editable mode:

```bash
pip install -e .
pip install pycocotools
```

If you only need CPU debugging, install the CPU version of PyTorch instead. For paper-scale experiments, use a CUDA environment consistent with your local driver and GPU.

## Data Preparation

The experiments use COCO-style detection annotations. A typical dataset layout is:

```text
RUOD/
├─ images/
│  ├─ train/
│  └─ val/
└─ annotations/
   ├─ instances_train.json
   └─ instances_val.json
```

Update your dataset YAML so that `train`, `val`, and annotation paths point to your local files. The released code searches several common annotation locations, including:

```text
{data_path}/annotations/instances_val.json
{data_path}/annotations/instances_{split}.json
{data_path}/val/_annotations.coco.json
{data_path}/instances_val.json
{data_path}/_annotations.coco.json
```

The RUOD dataset can be obtained from [Baidu AI Studio](https://aistudio.baidu.com/datasetdetail/216919).

## Run Training

The simplest way is to edit `ultralytics/train.py` for your local model and dataset paths, then run:

```bash
python ultralytics/train.py
```

Minimal API example:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8s.yaml")

results = model.train(
    data="path/to/data.yaml",
    epochs=250,
    imgsz=640,
    seed=0,
    deterministic=True,
    save_json=True,
    use_coco_fitness=True,
    coco_eval_interval=5,
    coco_only_best=True,
    coco_start_epoch=100,
    patience=100,
)
```

Recommended CDP settings:

| Argument | Recommended value | Meaning |
| --- | --- | --- |
| `save_json` | `True` | Save predictions for COCO API evaluation. |
| `use_coco_fitness` | `True` | Use COCO AP as the checkpoint-selection metric. |
| `coco_eval_interval` | `5` or `10` | Run COCO API periodically to reduce overhead. |
| `coco_only_best` | `True` | Only COCO-evaluated epochs may update `best.pt`. |
| `coco_start_epoch` | warm-up epoch | Skip early unstable epochs if needed. |

For a quick pipeline check, disable COCO fitness:

```python
save_json=False
use_coco_fitness=False
```

## Outputs

Typical training outputs are saved under `runs/`:

```text
runs/
└─ detect/
   └─ train*/
      ├─ weights/
      │  ├─ best.pt
      │  └─ last.pt
      ├─ predictions.json
      └─ results.csv
```

When CDP settings are enabled, `best.pt` is selected from epochs that actually run COCO API evaluation.

## Visualization

Qualitative and plotting utilities are included for checking model behavior and preparing figures:

```bash
python ultralytics/visual.py
python ultralytics/plotfig2.py
```

Adjust input image, weight, and output paths inside the scripts according to your local experiment directory.

## Acknowledgements and License

This repository is a research-oriented derivative implementation based on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). The upstream copyright notices and GNU AGPL-3.0 license are retained. See [LICENSE](LICENSE).
