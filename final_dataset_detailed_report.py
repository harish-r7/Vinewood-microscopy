import os
from collections import Counter, defaultdict
import json

def analyze_split(split_dir):
    images_path = os.path.join(split_dir, 'images')
    masks_path = os.path.join(split_dir, 'masks')
    
    imgs_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith('.png')])
    msks_files = sorted([f for f in os.listdir(masks_path) if f.lower().endswith('.png')])
    
    match = len(imgs_files) == len(msks_files) and imgs_files == msks_files
    
    img_counter = Counter()
    msk_counter = Counter()
    for f in imgs_files:
        if 'with' in f:
            img_counter['with'] += 1
        elif 'wout' in f:
            img_counter['wout'] += 1
        if 'real' in f:
            img_counter['real'] += 1
        elif 'gan' in f:
            img_counter['gan'] += 1
    
    for f in msks_files:
        if 'with' in f:
            msk_counter['with'] += 1
        elif 'wout' in f:
            msk_counter['wout'] += 1
        if 'real' in f:
            msk_counter['real'] += 1
        elif 'gan' in f:
            msk_counter['gan'] += 1
    
    return {
        'total_images': len(imgs_files),
        'total_masks': len(msks_files),
        'filenames_match': match,
        'images_categories': dict(img_counter),
        'masks_categories': dict(msk_counter),
        'sample_images': imgs_files[:10],
        'sample_masks': msks_files[:10]
    }

root = os.path.join(os.getcwd(), 'final_dataset')
report = {}
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(root, split)
    if os.path.exists(split_dir):
        report[split] = analyze_split(split_dir)

total_imgs = sum(d['total_images'] for d in report.values())
total_msks = sum(d['total_masks'] for d in report.values())

summary = {
    'report': report,
    'total_images': total_imgs,
    'total_masks': total_msks,
    'all_splits_match': all(d['filenames_match'] for d in report.values())
}

print(json.dumps(summary, indent=2))

