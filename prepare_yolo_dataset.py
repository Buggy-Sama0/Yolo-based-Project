
# COCO to YOLO conversion function
import os
import json
from collections import defaultdict

def prepare_yolo_dataset(coco_json_path, images_dir, labels_dir):
    """
    Convert COCO-format annotation JSON to YOLO-format .txt labels.
    Args:
        coco_json_path: Path to COCO annotation JSON file
        images_dir: Directory containing images
        labels_dir: Output directory for YOLO labels
    """
    os.makedirs(labels_dir, exist_ok=True)
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    img_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    img_id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}
    cats = sorted(coco['categories'], key=lambda x: x['id'])
    cat_id_to_index = {cat['id']: idx for idx, cat in enumerate(cats)}

    img_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    for img_id, anns in img_to_anns.items():
        filename = img_id_to_filename[img_id]
        width, height = img_id_to_size[img_id]
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')
        with open(label_path, 'w') as f:
            for ann in anns:
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                class_id = cat_id_to_index[ann['category_id']]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    print(f"YOLO labels written to {labels_dir}")
