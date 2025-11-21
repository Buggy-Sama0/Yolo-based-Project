import os
import requests
import zipfile


def download_coco():
    # Create directories
    # os.makedirs('images/val', exist_ok=True)
    os.makedirs('images/train', exist_ok=True)
    # Download train images (20GB - be careful!
    # Only download if you have space, otherwise use val set first
    # os.system(f"wget {train_url} -O coco/train2017.zip")
    
    # Download val images (1GB - recommended for testing)
    # val_url = "http://images.cocodataset.org/zips/val2017.zip"
    train_url = "http://images.cocodataset.org/zips/train2017.zip"
    print("Downloading train images...")
    response=requests.get(train_url, stream=True)
    filename=train_url.split('/')[-1]
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

    
    # Extract files
    print("Extracting images...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('images/train')
    print("Images extracted to images/train")
    # with zipfile.ZipFile('coco/train2017.zip', 'r') as zip_ref:
    #     zip_ref.extractall('coco/images/')
    
    os.remove(filename)
    print("Zip file removed.")


def download_coco_annotations():
    os.makedirs('annotations', exist_ok=True)
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = "annotations_trainval2017.zip"

    print("Downloading COCO annotations...")
    response = requests.get(ann_url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

    print("Extracting annotation files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("annotations")
    print("Annotations extracted to ./annotations/")

    os.remove(zip_path)
    print("Zip file removed.")

#download_coco()  # Uncomment to run
download_coco_annotations()