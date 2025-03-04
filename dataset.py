import json
import os 
from tqdm import tqdm
import shutil
import random
from sklearn.model_selection import train_test_split

def duplicate_and_split_dataset(parent_dir, num_duplicates=50, train_size=0.7, val_size=0.15):
    images_dir = os.path.join(parent_dir, "images")
    labels_dir = os.path.join(parent_dir, "labels")
    image_names = os.listdir(images_dir)
    
    # Create dataset structure
    dataset_root = os.path.join(parent_dir, "processed_dataset")
    splits = ['train', 'valid', 'test']
    
    # Create directories for each split
    for split in splits:
        os.makedirs(os.path.join(dataset_root, split), exist_ok=True)

    # Collect all samples
    all_samples = []
    for image in tqdm(image_names, desc="Reading original samples", colour="red"):
        label_path = os.path.join(labels_dir, f"{image}.jsonl")
        with open(label_path, "r") as f:
            label = json.load(f)
        
        sample = {
            "image": image,
            "prefix": "extract data in JSON format",
            "suffix": {}
        }
        
        for obj in label["objects"]:
            class_name = obj["classTitle"]
            text = obj["description"]
            if class_name in sample["suffix"]:
                sample["suffix"][class_name].append(text)
            else:
                sample["suffix"][class_name] = [text]
        
        # Duplicate samples
        for i in range(num_duplicates):
            new_sample = sample.copy()
            new_image_name = f"{image.split('.')[0]}_{i}{os.path.splitext(image)[1]}"
            new_sample["image"] = new_image_name
            all_samples.append((new_sample, image))

    # Split samples
    train_val_samples, test_samples = train_test_split(all_samples, test_size=1-train_size-val_size, random_state=42)
    train_samples, val_samples = train_test_split(train_val_samples, test_size=val_size/(train_size+val_size), random_state=42)

    # Process splits
    splits_data = {
        'train': train_samples,
        'valid': val_samples,
        'test': test_samples
    }

    for split_name, samples in splits_data.items():
        print(f"Processing {split_name} split ({len(samples)} samples)")
        annotations_file = os.path.join(dataset_root, split_name, "annotations.json")
        
        with open(annotations_file, "w") as f_out:
            for sample, orig_image in tqdm(samples, desc=f"Creating {split_name} dataset", colour="green"):
                # Copy and rename image
                src_image = os.path.join(images_dir, orig_image)
                dst_image = os.path.join(dataset_root, split_name, sample["image"])
                shutil.copy2(src_image, dst_image)
                # Write annotation
                f_out.write(json.dumps(sample) + "\n")

    print(f"\nDataset created successfully in {dataset_root}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

if __name__ == "__main__":
    parent_dir = "./tested_claims"
    duplicate_and_split_dataset(parent_dir)