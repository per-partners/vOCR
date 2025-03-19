import json
import os
import re
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Dict, List

# Standardizing class names
CLASSES = {
    "medical provider name": ["Medical Provider", "medical provider"],
    "national id": ["national id", "National ID"],
    "employer name": ["employer name", "Employer Name"],
    "mobile number": ["mobile number", "Mobile Number"],
    "claim ID": ["Digital", "digital"],
    "stamp": ["STAMPS", "Stamps", "stamp"],
    "name": ["name", "Name", "NAMES"],
    "drug": ["Drug", "drugs"],
    "general handwritten": ["Hand Written", "general handwritten", "WRITTEN"],
    "number": ["Number", "number"],
    "unknown": ["UNK", "unknown"],
    "medical information": ["age", "address", "Approval No"],
    "diagnosis": ["DIAGNOSIS", "diagnosis"],
    "test": ["TESTS", "Tests", "test"],
    "scan": ["scan", "Scan"],
    "signature": ["doctor signature", "Signature"],
    "instructions": ["instructions", "Instruction"],
    "drug instructions": ["Drug Instructions"],
    "price": ["Price", "price"],
    "date": ["date", "Date"],
    "dental symbol": ["dental symbol plus", "dental symbol T"],
    "medical id": ["Medical ID", "medical ID"],
}

# Normalize class names
NORMALIZED_CLASSES: Dict[str, str] = {
    variant.lower(): canonical
    for canonical, variants in CLASSES.items()
    for variant in variants
}


def normalize_class_title(class_title: str) -> str:
    return NORMALIZED_CLASSES.get(class_title.lower(), class_title)


def is_valid_claim_id(claim_id: str) -> bool:
    return bool(re.fullmatch(r"^[a-zA-Z0-9]+$", claim_id))


def duplicate_and_split_dataset(parent_dir: str, train_size: float = 0.7, seed: int = 42):
    images_dir = os.path.join(parent_dir, "images")
    labels_dir = os.path.join(parent_dir, "annotation")
    dataset_root = os.path.join(parent_dir, "processed_dataset")
    os.makedirs(dataset_root, exist_ok=True)

    # Ensure images and annotations exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("Error: Images or annotation directory is missing!")
        return

    image_names = os.listdir(images_dir)
    train_images, valid_images = train_test_split(
        image_names, test_size=1 - train_size, random_state=seed)
    splits = {"train": train_images, "valid": valid_images}

    for split_name, image_list in splits.items():
        split_dir = os.path.join(dataset_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        annotations_file = os.path.join(split_dir, "annotations.jsonl")

        with open(annotations_file, "w") as f_out:
            for count, image_name in enumerate(tqdm(image_list, 
                                                    desc=f"Processing {split_name}", 
                                                    colour="green")):
                label_path = os.path.join(labels_dir, f"{image_name}.json")
                image_path = os.path.join(images_dir, image_name)

                if not os.path.exists(label_path) or not os.path.exists(image_path):
                    print(
                        f"Warning: Missing label or image for {image_name}, skipping.")
                    continue

                with open(label_path, "r") as f:
                    label = json.load(f)

                image = cv2.imread(image_path)
                if image is None:
                    print(
                        f"Warning: Unable to read image {image_name}, skipping.")
                    continue

                sample = {
                    "image_name": f"{count:06d}.jpg",
                    "prefix": "Describe the handwritten text in the image in JSON Format",
                    "suffix": {},
                }

                skip_image = False
                for obj in label.get("objects", []):
                    class_name = normalize_class_title(obj["classTitle"])
                    description = obj.get("description", "")

                    if class_name == "Claim ID" and not is_valid_claim_id(description):
                        continue

                    if class_name == "unknown":
                        skip_image = True
                        break

                    sample["suffix"].setdefault(
                        class_name, []).append(description)

                if skip_image:
                    continue

                dst_image_path = os.path.join(split_dir, sample["image_name"])
                cv2.imwrite(dst_image_path, image)
                f_out.write(json.dumps(sample) + "\n")

    print(f"\nDataset successfully created in {dataset_root}")


if __name__ == "__main__":
    parent_directory = "./assets/batch_claims"
    duplicate_and_split_dataset(parent_directory)
