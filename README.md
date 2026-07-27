# Skin Lesion Segmentation

Task 1 skin-lesion segmentation project with six binary segmentation models:

- `unet` - CNN baseline.
- `resnet34_unet` - ImageNet-pretrained ResNet-34 encoder with U-Net decoder.
- `resnet50_unet` - ImageNet-pretrained ResNet-50 encoder with U-Net decoder.
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

Available values are `unet`, `resnet34_unet`, `resnet50_unet`, `lb_unet`, `segformer_b1`, and `uctransnet`.

2. Run training:

```powershell
python train_task1.py
```

Each model's batch size, learning rate, checkpoint path, and optional pretrained setting live under the `models` section in `settings.json`. Any model with both `pretrained: true` and an encoder (ResNet-U-Net and SegFormer-B1) uses staged fine-tuning: its encoder is frozen for `freeze_encoder_epochs`, then updated with `encoder_learning_rate`, while the rest of the model uses `learning_rate`. Training uses automatic mixed precision on CUDA.

The global `training.loss` block controls the primary objective. By default it is `1.0 * BCE + 1.0 * Dice`. Set `bce_weight_decay.enabled` to `true` to linearly decay the primary BCE coefficient toward `target_weight` over `decay_epochs`; it starts when a pretrained encoder is unfrozen, or at epoch 1 for non-pretrained models. LB-UNet's PMA auxiliary supervision is intentionally unaffected.

Each model's checkpoint path is explicit in `settings.json` (for example, `models.lb_unet.checkpoint_path`). The default paths are:

```text
checkpoints/<model_name>/task1_<model_name>.pt
outputs/task1/training/<model_name>/curves.png
outputs/task1/predictions/<model_name>/
```

`training.variant_sampling` controls how fixed Task 1 augmentations are used.
With `"one_per_source"` (the default), each epoch randomly selects one of the
ten prepared variants for every source image; use `"all_variants"` to train on
every fixed variant in each epoch. Validation is unchanged.

## Train Task 2 attributes

Task 2's primary experiment is RGB-only with Task 1 prediction-guided ROI
cropping. Select either `task2_resnet34_multidecoder_roi` or
`task2_segformer_b1_multidecoder_roi` in
`settings.json` under `task2.model_name`. Both models share one encoder
initialized from the matching Task 1 checkpoint, then use five complete,
parameter-independent decoders (one per attribute).

Then train:

```powershell
python train_task2.py
```

The ROI profiles use the largest connected component in the Task 1 predicted
masks in `data/prepared/task2/*/mask` to crop a square lesion region (10%
safety margin), resize it to 256x256, and
restore validation predictions to the complete 256x256 canvas. The mask is
only used to locate the crop: it is not supplied to the model as a fourth input
channel. Empty predictions and predicted boxes below 32 pixels are skipped.
Generate the train and validation masks with the same Task 1 checkpoint before
training:

```powershell
python prepare_task2_priors.py --model segformer_b1 --split both
```

The selected Task 2 profile transfers only the matching Task 1 encoder. It
accepts an RGB image and outputs five independent attribute logits (each is
passed through sigmoid independently). Each active attribute uses equal-coefficient BCE and
Focal Tversky loss; `task2.loss.attribute_loss` sets its initial weight and its
own Tversky parameters. The dynamic policy reduces a stagnant attribute's loss
weight, then restores it when validation Dice improves.

`task2.training.variant_sampling` controls fixed augmentation sampling:

```json
"variant_sampling": "one_per_source"
```

With `one_per_source`, each epoch randomly selects one of the ten fixed
augmentations for every original image (about 2,430 training samples per epoch).
Use `all_variants_weighted` to retain the earlier 24,300-sample weighted variant
sampler. Validation is never augmented or randomly sampled.

The configured checkpoint path is the validation-best model; the trainer also
writes `<checkpoint_name>_latest.pt` after every epoch. Task 1 settings and
training remain independent.

The non-ROI profiles remain available as an RGB-only full-image comparison.
A future RGB+mask four-channel model should use a separate profile, because the
extra channel changes the pretrained encoder input distribution.

## Inference and evaluation

Run inference with the model currently selected in `settings.json`:

```powershell
python infer_task1.py
```

Optional explicit paths:

```powershell
python infer_task1.py --checkpoint checkpoints/uctransnet/task1_uctransnet.pt --output outputs/task1/predictions/uctransnet
```

Compare every configured checkpoint on the original, unaugmented validation set defined by `splits/task1_val.csv`:

```powershell
python evaluate_task1.py
```

The evaluator writes `validation_model_comparison.csv` with hard Dice, confidence Dice (soft Dice from probabilities), HD, and HD95. It also runs every model on the folder in `evaluation.sample_input` and writes masks plus GT/prediction comparison images under `outputs/task1/evaluation/sample_predictions/<model_name>/`.

To evaluate selected models or another sample folder:

```powershell
python evaluate_task1.py --models lb_unet segformer_b1 --sample-input path\to\sample_images --sample-ground-truth path\to\sample_masks
```

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
