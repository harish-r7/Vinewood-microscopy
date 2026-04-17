import os
import json
from collections import Counter

root = os.path.join(os.getcwd(), 'dataset')
report = {}
for split in sorted(os.listdir(root)):
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        continue
    report[split] = {}
    for sub in sorted(os.listdir(split_dir)):
        subdir = os.path.join(split_dir, sub)
        if not os.path.isdir(subdir):
            continue
        exts = Counter()
        names = []
        total = 0
        for dirpath, _, files in os.walk(subdir):
            for f in sorted(files):
                total += 1
                ext = os.path.splitext(f)[1].lower() or '<none>'
                exts[ext] += 1
                names.append(f)
        report[split][sub] = {
            'total_files': total,
            'extensions': dict(exts),
            'sample_names': names[:5],
        }

mismatch = {}
for split, data in report.items():
    imgs = data.get('images', {}).get('total_files', 0)
    mks = data.get('masks', {}).get('total_files', 0)
    mismatch[split] = {'images': imgs, 'masks': mks, 'difference': imgs - mks}

print(json.dumps({'report': report, 'mismatch': mismatch}, indent=2))
