# Skin Lesion Segmentation

Task 1 skin-lesion segmentation project with four binary segmentation models:

- `unet` - CNN baseline.
- `lb_unet` - lightweight boundary-assisted U-Net with GSA and PMA auxiliary heads.
- `segformer_b1` - ImageNet-pretrained MiT-B1 SegFormer.
- `uctransnet` - U-Net with channel-wise Transformer skip fusion.

All models accept RGB images and return one logit map per image. The task is binary lesion segmentation.

## Environment

This project is intended to run in the existing `IC_summer` Conda environment.

```powershell
conda activate IC_summer
python -m pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build separately before installing the requirements if PyTorch is not already working. Do not run a generic PyTorch installation command that replaces a working CUDA build.

Verify the environment:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`SegFormer-B1` downloads its ImageNet MiT-B1 weights on the first training run when `models.segformer_b1.pretrained` is `true` in `settings.json`.

## Data layout

Keep the original data unchanged:

```text
data/train/
|- images/
|  `- 000001.jpg
|- task1_gt/
|  `- 000001_segmentation.png
`- task2_gt/
   `- 000001_attribute_<attribute>.png
```

The preprocessing command creates fixed augmented datasets without modifying `data/train`:

```text
data/prepared/
|- task1/
|  |- train/images/
|  |- train/task1_gt/
|  |- val/images/
|  `- val/task1_gt/
`- task2/
   |- train/images/
   |- train/task2_gt/
   |- val/images/
   `- val/task2_gt/
```

## Prepare fixed augmented data

Run this once before training:

```powershell
python data_preprocessing.py prepare-training
```

The command audits the source data, creates a fixed train/validation split, and generates ten Task 1 and Task 2 training variants per source image. Validation samples receive only deterministic resize and padding.

The fixed variants are: base preprocessing, horizontal flip, vertical flip, affine transform, brightness/contrast, HSV, CLAHE, Gaussian noise, Gaussian blur, and image compression. Only geometric transforms modify masks; colour and degradation transforms leave masks unchanged.

## Train a model

1. Open `settings.json` and set the model name:

```json
"model_name": "uctransnet"
```

Available values are `unet`, `lb_unet`, `segformer_b1`, and `uctransnet`.

2. Run training:

```powershell
python train_task1.py
```

Each model's batch size, learning rate, and optional pretrained setting live under the `models` section in `settings.json`. Training uses automatic mixed precision on CUDA.

Outputs are kept separate by model:

```text
checkpoints/<model_name>/best.pt
outputs/task1/training/<model_name>/curves.png
outputs/task1/predictions/<model_name>/
```

## Inference and evaluation

Run inference with the model currently selected in `settings.json`:

```powershell
python infer_task1.py
```

Optional explicit paths:

```powershell
python infer_task1.py --checkpoint checkpoints/uctransnet/best.pt --output outputs/task1/predictions/uctransnet
```

Evaluate predicted masks against Task 1 ground truth:

```powershell
python evaluate_task1.py --predictions outputs/task1/predictions/uctransnet --masks data/train/task1_gt
```

The evaluator reports mean Dice, Hausdorff distance, and HD95.

## Useful preprocessing commands

```powershell
# Audit Task 1 or Task 2 source data
python data_preprocessing.py audit --task 1
python data_preprocessing.py audit --task 2

# Create only a split manifest
python data_preprocessing.py split --task 1

# Preview random paired augmentations
python data_preprocessing.py preview --task 1 --image-id 000001
```

## Git repository guidance

Commit source code, `settings.json`, `requirements.txt`, and this README. Do not commit raw data, prepared augmented data, checkpoints, or output images. Add these paths to `.gitignore` before the first commit:

```gitignore
data/
checkpoints/
outputs/
splits/
__pycache__/
*.py[cod]
.vscode/
```
