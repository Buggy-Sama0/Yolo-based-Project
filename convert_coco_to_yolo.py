import os
import json
from tqdm import tqdm
from PIL import Image

# Paths
COCO_ANN_PATH = 'annotations_trainval2017/annotations/instances_val2017.json'  # Path to COCO val annotation
IMAGES_DIR = 'images/val'  # Directory with validation images
LABELS_DIR = 'labels/val'  # Output directory for YOLO labels

os.makedirs(LABELS_DIR, exist_ok=True)

# Load COCO annotations
with open(COCO_ANN_PATH, 'r') as f:
    coco = json.load(f)

# Build image id to filename mapping
img_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
img_id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}

# Build category id to index mapping (YOLO expects 0-based class ids)
cats = sorted(coco['categories'], key=lambda x: x['id'])
cat_id_to_index = {cat['id']: idx for idx, cat in enumerate(cats)}

# Group annotations by image
from collections import defaultdict
img_to_anns = defaultdict(list)
for ann in coco['annotations']:
    img_to_anns[ann['image_id']].append(ann)

for img_id, anns in tqdm(img_to_anns.items()):
    filename = img_id_to_filename[img_id]
    width, height = img_id_to_size[img_id]
    label_path = os.path.join(LABELS_DIR, os.path.splitext(filename)[0] + '.txt')
    with open(label_path, 'w') as f:
        for ann in anns:
            # COCO bbox: [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            class_id = cat_id_to_index[ann['category_id']]
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
print(f"YOLO labels written to {LABELS_DIR}")
