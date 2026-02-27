# SAM-Fine-Tuning-for-Medical-Imaging-

# SAM Fine-Tuning on Kvasir-SEG
### Medical Image Segmentation — Polyp Detection

---

## Overview

This project fine-tunes **Segment Anything Model (SAM)** on the **Kvasir-SEG** dataset for gastrointestinal polyp segmentation. It implements a transfer learning pipeline with a frozen encoder and a lightweight trainable decoder, running on Kaggle with GPU support.

The architecture supports two encoder backends:
- **SAM ViT-B** (primary) — frozen image encoder from Meta's Segment Anything Model
- **ResNet-50** (fallback) — ImageNet-pretrained, used when the SAM checkpoint is unavailable

---

## Dataset

**Kvasir-SEG** — A public dataset of gastrointestinal polyp images with pixel-level segmentation masks.

- **Source:** [Kaggle — debeshjha1/kvasirseg](https://www.kaggle.com/datasets/debeshjha1/kvasirseg)
- **Size:** 1,000 paired image/mask samples
- **Input resolution:** 512×512 px
- **Split:** 80% train / 10% val / 10% test

---

## Architecture

```
Input Image (3×512×512)
        ↓
[ Frozen Encoder ]
  SAM ViT-B  OR  ResNet-50 (layer4)
        ↓
[ Trainable Decoder — SimpleDecoder ]
  Conv → BN → ReLU
  Conv → BN → ReLU
  ConvTranspose2d (upsample ×2)
  Conv 1×1 → binary mask output
        ↓
Bilinear upsample → (1×512×512)
        ↓
Predicted Binary Mask
```

**Key design choices:**
- Encoder is **fully frozen** — only the decoder is trained
- Decoder is a lightweight UNet-style head (~256 channels)
- Output is a single-channel binary segmentation map

---

## Setup

### Requirements

```bash
pip install segment-anything
pip install torch torchvision
pip install scikit-learn pillow matplotlib kagglehub
```

### SAM Checkpoint

Download the ViT-B checkpoint from Meta:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O /kaggle/working/sam_vit_b.pth
```

> If the checkpoint is not found, the pipeline automatically falls back to a ResNet-50 encoder.

---

## Usage

Run the notebook cells in order on Kaggle:

| Cell | Description |
|------|-------------|
| 1 | Download Kvasir-SEG via `kagglehub` |
| 2–3 | Explore dataset structure |
| 4 | Build train/val/test splits and copy files |
| 5 | Visualize a sample image/mask pair |
| 6 | Define `KvasirSegDataset` and `DataLoader` |
| 7 | Build encoder-decoder model (SAM or ResNet fallback) |
| 8+ | Training loop, evaluation, metrics |

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Image size | 512×512 |
| Batch size | 8 |
| Train/Val/Test split | 80/10/10 |
| Encoder | SAM ViT-B (frozen) |
| Decoder channels | 256 |
| Random seed | 42 |

---

## Project Structure

```
A4_SAM_Fine_Tunning.ipynb   # Main notebook
/kaggle/working/
  kvasirseg_splits/
    images/train|val|test/
    masks/train|val|test/
  sam_vit_b.pth             # SAM checkpoint (download separately)
```

---

## References

- [Segment Anything — Meta AI](https://github.com/facebookresearch/segment-anything)
- [Kvasir-SEG Dataset Paper](https://arxiv.org/abs/1911.07069)
- [SAM Paper — Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)
