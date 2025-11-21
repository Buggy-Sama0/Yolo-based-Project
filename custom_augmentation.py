import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_custom_augmentations():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),
        A.ToFloat(max_value=255),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    return train_transform
