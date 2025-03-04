import json
import os
from tqdm import tqdm
import shutil
import random
from sklearn.model_selection import train_test_split
import cv2
import numpy as np 


def duplicate_and_split_dataset(parent_dir, train_size=0.7, seed=42):
    images_dir = os.path.join(parent_dir, "images")
    labels_dir = os.path.join(parent_dir, "annotation")
    dataset_root = os.path.join(parent_dir, "processed_dataset")

    splits = ["train", "valid"]

    # Create directories for each split
    for split in splits:
        os.makedirs(os.path.join(dataset_root, split), exist_ok=True)

    # get the image names list
    image_names = os.listdir(images_dir)
    count = 1
    # Collect all samples
    for split, split_name in tqdm(
        zip(train_test_split(image_names, test_size=1 - train_size, random_state=seed), splits),
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
                "prefix": "extract data in JSON format",
                "suffix": {},
            }

            do_we_need_to_skip_the_whole_image = False
            for obj in label["objects"]:
                class_name = obj["classTitle"]
                if class_name == "UNK":
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

            ## Crop Level
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
                if class_name == "UNK":
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
                    "image": image[bbox[1] : bbox[3], bbox[0] : bbox[2]],
                    "prefix": "Describe the handwritten text in the image in JSON Format.",
                    "suffix": {},
                }

                sample_crop["suffix"][class_name] = [obj["description"]]
                all_samples.append(sample_crop)
    
        with open(f"{os.path.join(dataset_root, split_name)}/annotations.jsonl", "w") as f_out:
            for sample in tqdm(
                all_samples, desc=f"Creating {split_name} dataset", colour="blue"
            ):
                # Copy and rename image
                dst_image = os.path.join(dataset_root, split_name, count)
                count +=1
                cv2.imwrite(dst_image, sample["image"])
                # Write annotation
                # remove the image object from the dict
                sample.pop("image")
                f_out.write(json.dumps(sample) + "\n")

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
