# CSI-Bench Reproduction Report

Reproduction of **CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing**, with a focus on **single-task supervised baselines**, reproducibility, and careful validation of data and training pipelines.

## Highlights

- Reproduced multiple single-task baselines from the public CSI-Bench release
- Identified and corrected inconsistencies between the public default setup and the task definitions reported in the paper
- Diagnosed issues in label mappings, metadata formats, split behavior, and dataset organization
- Obtained high-fidelity results on key tasks that are highly consistent with the paper

## Covered Tasks

The reproduced single-task supervised tasks include:

- **Motion Source Recognition (MSR)** — 4-class
- **Room-Level Localization** — 6-class
- **Breathing Detection** — 2-class

## Main Results

### Motion Source Recognition (MSR)

- **ResNet18**: **99.63 / 99.63** (Accuracy / F1)  
  High-fidelity reproduction, highly consistent with the paper.

- **Transformer**: **95.08 / 95.10**  
  Valid reproduction, but still with a noticeable gap from the reported result.

### Room-Level Localization

- **Transformer (test_id)**: **99.27 / 99.27**  
  High-fidelity reproduction, essentially matching the paper's main reported result.

### Breathing Detection

- **Transformer**
  - test_easy_id: **87.68 / 87.85**
  - test_hard_id: **84.92 / 84.56**
  - test: **85.31 / 85.21**
  - test_medium_id: **82.61 / 82.22**

## Result Summary

| Task | Model | Result | Status |
|---|---|---:|---|
| Motion Source Recognition | ResNet18 | 99.63 / 99.63 | High-fidelity |
| Motion Source Recognition | Transformer | 95.08 / 95.10 | Valid, with gap |
| Room-Level Localization | Transformer | 99.27 / 99.27 | High-fidelity |
| Breathing Detection | Transformer | 85.31 / 85.21 | Valid |

## Technical Notes

Several non-trivial issues had to be resolved during reproduction:

- the public default MSR label mapping was binary, while the paper defines MSR as a 4-class task
- `Localization` metadata required a compatibility fix by adding `num_classes`
- under the current public data and metadata combination, `Localization` easy/medium splits are not fully compatible
- `BreathingDetection` contains a small number of empty H5 files, which can be skipped without blocking full training

## Environment

- Python 3.10
- PyTorch 2.1.2
- CUDA 11.8
- 1 × 48GB GPU
- 25 vCPU
- 92GB RAM

## Data Layout

```text
CSI_MultiTask_Project/
├── data/
│   ├── csi-bench-dataset/
│   │   └── csi-bench-dataset/
│   │       ├── MotionSourceRecognition/
│   │       ├── Localization/
│   │       ├── BreathingDetection/
│   │       ├── HumanActivityRecognition/
│   │       ├── HumanIdentification/
│   │       ├── ProximityRecognition/
│   │       └── dataset-metadata.json
│   └── tasks/
│       ├── MotionSourceRecognition -> ...
│       ├── Localization -> ...
│       └── BreathingDetection -> ...
└── utils/
    └── CSI-Bench-Real-WiFi-Sensing-Benchmark-main/
```

## Example Commands

### Motion Source Recognition (ResNet18, 4-class)
```text
python scripts/train_supervised.py \
    --data_dir="/root/autodl-tmp/CSI_MultiTask_Project/data" \
    --task_name="MotionSourceRecognition" \
    --model="resnet18" \
    --batch_size=32 \
    --epochs=100 \
    --win_len=500 \
    --feature_size=232 \
    --save_dir="./results_fourclass" \
    --output_dir="./results_fourclass" \
    --num_workers=0 \
    --test_splits="all" \
    --learning_rate=0.0001
```
    
### Room-Level Localization (Transformer)
```text
python scripts/train_supervised.py \
    --data_dir="/root/autodl-tmp/CSI_MultiTask_Project/data" \
    --task_name="Localization" \
    --model="transformer" \
    --batch_size=32 \
    --epochs=100 \
    --win_len=500 \
    --feature_size=232 \
    --save_dir="./results_localization_testid" \
    --output_dir="./results_localization_testid" \
    --num_workers=0 \
    --test_splits="test_id" \
    --learning_rate=0.0001
```
    
### Breathing Detection (Transformer)
```text
python scripts/train_supervised.py \
    --data_dir="/root/autodl-tmp/CSI_MultiTask_Project/data" \
    --task_name="BreathingDetection" \
    --model="transformer" \
    --batch_size=32 \
    --epochs=100 \
    --win_len=300 \
    --feature_size=232 \
    --save_dir="./results_breathing" \
    --output_dir="./results_breathing" \
    --num_workers=0 \
    --test_splits="all" \
    --learning_rate=0.0001
```
    
## Summary

This repository documents a structured reproduction of CSI-Bench single-task baselines. In particular:

- Motion Source Recognition (ResNet18)
- Room-Level Localization (Transformer)

achieved high-fidelity results that are highly consistent with the paper.

At the same time, Motion Source Recognition (Transformer) and Breathing Detection (Transformer) were also successfully reproduced, forming a reasonably complete set of single-task benchmark results.
