# Final Dataset Report (Detailed Matching Analysis)

## Overview
- Total images: 2000
- Total masks: 2000
- **Filenames perfectly match across images and masks in all splits**
- All splits balanced and verified identical pairing
- All files are `.png`
- **All splits match: true**

## Split Breakdown

### Train (1400 images/masks - perfect match)
- Images/masks categories: with=700, wout=700, real=619, gan=480

### Validation (300 images/masks - perfect match)
- Images/masks categories: with=150, wout=150, real=131, gan=108

### Test (300 images/masks - perfect match)
- Images/masks categories: with=150, wout=150, real=131, gan=104

## Category Summary (Across Images/Masks)
| Split | with | wout | real | gan |
|-------|------|------|------|-----|
| train | 700 | 700 | 619 | 480 |
| val   | 150 | 150 | 131 | 108 |
| test  | 150 | 150 | 131 | 104 |
| **Total** | **1000** | **1000** | **881** | **692** |

## Notes
- **Perfect filename matching confirmed**: Every image has exact corresponding mask by name in each split.
- Mix of real (real_*) and GAN-generated (gan_*) samples.
- Sample filenames:
  - Train: `gan_wout_0000.png` ...
  - Val: `gan_with_0000.png`, `gan_wout_0584.png` ...
  - Test: `gan_wout_0480.png` ..., `real_with_000000.png`, `real_wout_000085.png` ...
- Run `python final_dataset_detailed_report.py` for full JSON with top samples.
- Analysis script: `final_dataset_detailed_report.py`
