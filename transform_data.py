import json
import os
from tqdm import tqdm
import shutil
import random
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import json
import re
from typing import AnyStr, List, Optional
from collections import Counter

from charset_normalizer.utils import is_arabic
from pyarabic.normalize import normalize_searchtext


classes = {
    "medical provider name": [
        "Medical  Provider",
        "Medical Provider",
        "medical provider"
    ],
    
    "national id": [
        "national id",
        "National ID"
    ],
    
    "employer name": [
        "employer name",
        "Employer Name"
    ],
    
    "mobile number": [
        "mobile number",
        "Mobile Number",
        "Mobile number"
    ],
    
    "claim ID": [
        "Digital",
        "digital"
    ],
    
    "stamp": [
        "STAMPS",
        "Stamps",
        "stamp",
        "Stamp",
        "STAMP"
    ],
    
    "name": [
        "name",
        "names",
        "Names",
        "Name",
        "NAMES"
    ],
    
    "drug": [
        "Drug",
        "drug",
        "drugs",
        "Drugs",
        "DRUGS",
        "DRUG"
    ],
    
    "general handwritten": [
        "Hand Written",
        "H.W GENERAL",
        "H.w.general",
        "H.W general",
        "general handwritten",
        "genaral handwritten",
        "general handwirtten",
        "General Handwritten",
        "Genaral Handwritten",
        "Written",
        "General handwritten",
        "H.W",
        "WRITTEN",
        "H.w general",
        "WR",

    ],
    
    "number": [
        "Number",
        "Number",
        "number",
    ],
    
    "unknown": [
        "UNK",
        "unk",
        "Uknown",
        "Unknown",
        "unknown"
    ],
    
    "medical information": [
        "age",
        "Depart.",
        "depart",
        "address",
        "Approval No",
        "approval no",
        "approval no.",
        "depart",
        "depart.",
        "weight"
    ],
    
    "diagnosis": [
        "DIAGNOSIS",
        "diagnosis",
        "dign.",
        "Diagnose",
        "Diagnosis",
        "diganosis"
    ],
    
    "test": [
        "TESTS",
        "Tests",
        "tests",
        "Test",
        "test",
        "teste",
    ],
    
    "scan": [
        "scan",
        "Scan"
    ],
    
    "signature": [
        "doctor signsture",
        "doctor signature",
        "Doctor Signature",
        "doctor Signature",
        "doctor singnature",
        "Signature",
        "signature"
    ],
    
    "instructions": [
        "Instructios",
        "Instrutions",
        "Insturctions",
        "instructions",
        "INSTRUCTIONS",
        "instruction",
        "insturctions",
        "inst",
        "Instructions",
        "Instruction",
        "instructons",
        "Instrutions",
    ],

    "drug instructions": [
        "Drug Instructions",
        "drug instructions",
        "Drug Insructions",
        "Drug Instructions",
        "Drug Instuctions",
    ],
    
    "price": [
        "Price",
        "price"
    ],

    "date": [
        "date",
        "dates",
        "Date",
        "DATES",
        "Dates"
    ],
    
    "dental symbol": [
        "dental symbol plus",
        "dental symbol u.l",
        "dental symbol u.r",
        "dental symbol l.l",
        "dental symbol l.r",
        "dental symbol T"
    ],
    
    "medical id": [
        "Medical ID",
        "medical ID"
    ]
}


# Preprocess classes to create a normalized lookup
normalized_classes = {}
for canonical, variants in classes.items():
    for variant in variants:
        normalized_classes[variant.lower()] = canonical


def normalize_class_title(class_title: str) -> str:
    """Normalize the class title to its canonical form."""
    return normalized_classes.get(class_title.lower(), class_title)


def check_for_valid_claim_id(claim_id: str) -> bool:
    """Check if the claim ID is valid."""
    # create regex pattern for claim ID mix between digits and letters
    pattern = re.compile(r"^[a-zA-Z0-9]*$")
    return bool(pattern.match(claim_id))

def duplicate_and_split_dataset(parent_dir, num_duplicates=50, train_size=0.7, seed=42):
    images_dir = os.path.join(parent_dir, "images")
    labels_dir = os.path.join(parent_dir, "annotation")
    dataset_root = os.path.join(parent_dir, "processed_dataset")

    splits = ["train", "valid"]  # "test"]

    # Create directories for each split
    for split in splits:
        os.makedirs(os.path.join(dataset_root, split), exist_ok=True)

    # get the image names list
    image_names = os.listdir(images_dir)
    count = 1
    # Collect all samples
    for split, split_name in tqdm(
        zip(train_test_split(image_names, test_size=1 -
            train_size, random_state=seed), splits),
        desc="Splitting dataset",
        colour="red",
    ):
        all_samples = []
        for image_name in tqdm(split, desc="Processing samples", colour="green"):
            label_path = os.path.join(labels_dir, f"{image_name}.json")
            image_path = os.path.join(images_dir, image_name)

            # open json file
            with open(label_path, "r") as f:
                label = json.load(f)

            image = cv2.imread(image_path)

            # Claim level
            sample = {
                "image_name": image_name,
                "image": image,
                "prefix": "Describe the handwritten text in the image in JSON Format",
                "suffix": {},
            }

            do_we_need_to_skip_the_whole_image = False
            for obj in label["objects"]:
                class_name = obj["classTitle"]
                class_name = normalize_class_title(class_name)
                if class_name == "Claim ID":
                    # if the claim ID is not all digits, skip this object
                    # sometimes the claim id is mix between digits and letters so we need to take this
                    # into consideration                    
                    if check_for_valid_claim_id(obj["description"]):
                        continue
                    
                if class_name == "unknown":
                    do_we_need_to_skip_the_whole_image = True
                    break
                
                text = obj["description"]
                if class_name in sample["suffix"]:
                    sample["suffix"][class_name].append(text)
                else:
                    sample["suffix"][class_name] = [text]

            # save the original sample with its image
            if not do_we_need_to_skip_the_whole_image:
                all_samples.append(sample)

            # Crop Level
            for obj in label["objects"]:
                crop_name = (
                    image_name
                    + "_"
                    + obj["classTitle"]
                    + "_"
                    + obj["description"]
                    + ".jpg"
                )

                class_name = obj["classTitle"]
                class_name = normalize_class_title(class_name)
                if class_name == "unknown":
                    continue
                
                # if the claim ID is not all digits, skip this object
                # sometimes the claim id is mix between digits and letters so we need to take this
                # into consideration                    
                if check_for_valid_claim_id(obj["description"]):
                    continue
                
                geo_type = obj["geometryType"]
                
                bbox = obj["points"]["exterior"]
                if geo_type == "polygon":
                    # convert polygon to rectangle
                    points = obj["points"]["exterior"]
                    x = [point[0] for point in points]
                    y = [point[1] for point in points]
                    x_min, x_max = min(x), max(x)
                    y_min, y_max = min(y), max(y)
                    bbox = [x_min, y_min, x_max, y_max]

                bbox = np.array(bbox).flatten().tolist()

                sample_crop = {
                    "image_name": crop_name,
                    "image": image[bbox[1]: bbox[3], bbox[0]: bbox[2]],
                    "prefix": "Describe the handwritten text in the image in JSON Format.",
                    "suffix": {},
                }
                sample_crop["suffix"][class_name] = [obj["description"]]
                all_samples.append(sample_crop)

        with open(f"{os.path.join(dataset_root, split_name)}/annotations.jsonl", "w") as f_out:
            for sample in tqdm(
                all_samples, desc=f"Creating {split_name} dataset", colour="blue"
            ):
                # Saves images as '000001.jpg', '000002.jpg', etc.
                new_image_name = f"{count:06d}.jpg"
                dst_image = os.path.join(
                    dataset_root, split_name, new_image_name)
                cv2.imwrite(dst_image, sample["image"])
                sample["image_name"] = new_image_name
                sample.pop("image")
                f_out.write(json.dumps(sample) + "\n")
                count += 1

    # # Split samples
    # train_val_samples, test_samples = train_test_split(
    #     all_samples, test_size=1 - train_size - val_size, random_state=42
    # )
    # train_samples, val_samples = train_test_split(
    #     train_val_samples, test_size=val_size / (train_size + val_size), random_state=42
    # )

    # # Process splits
    # splits_data = {"train": train_samples, "valid": val_samples, "test": test_samples}
    # for split_name, samples in splits_data.items():
    #     print(f"Processing {split_name} split ({len(samples)} samples)")
    #     annotations_file = os.path.join(dataset_root, split_name, "annotations.json")

    #     with open(annotations_file, "w") as f_out:
    #         for sample, orig_image in tqdm(
    #             samples, desc=f"Creating {split_name} dataset", colour="green"
    #         ):
    #             # Copy and rename image
    #             src_image = os.path.join(images_dir, orig_image)
    #             dst_image = os.path.join(dataset_root, split_name, sample["image"])
    #             shutil.copy2(src_image, dst_image)
    #             # Write annotation
    #             f_out.write(json.dumps(sample) + "\n")

    # print(f"\nDataset created successfully in {dataset_root}")
    # print(f"Train samples: {len(train_samples)}")
    # print(f"Validation samples: {len(val_samples)}")


if __name__ == "__main__":
    parent_dir = "./assets/batch_claims"
    duplicate_and_split_dataset(parent_dir)
