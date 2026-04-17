# Dataset Report

## Overview
- Total images: 1308
- Total masks: 1308
- Image/mask counts are balanced for every split
- All dataset files are `.png`

## Split Breakdown

### Train
- `dataset/train/images`: 1046 files
- `dataset/train/masks`: 1046 files

### Validation
- `dataset/val/images`: 131 files
- `dataset/val/masks`: 131 files

### Test
- `dataset/test/images`: 131 files
- `dataset/test/masks`: 131 files

## With / Wout Counts
- `dataset/train/images`: `with` 826, `wout` 220
- `dataset/train/masks`: `with` 826, `wout` 220
- `dataset/val/images`: `with` 85, `wout` 46
- `dataset/val/masks`: `with` 85, `wout` 46
- `dataset/test/images`: `with` 85, `wout` 46
- `dataset/test/masks`: `with` 85, `wout` 46

## Notes
- Sample filename pattern: `real_with_000000.png`, `real_with_000001.png`, etc.
- GAN-generated images were removed from the dataset prior to this report.
