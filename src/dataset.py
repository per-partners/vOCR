import json
import os
import subprocess
from typing import Tuple

from PIL import Image
from qwen_vl_utils import process_vision_info
from roboflow import download_dataset
from torch.utils.data import DataLoader, Dataset

from src.utilities import format_data

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str, system_message):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()
        self.system_message = system_message

    def _load_entries(self):
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path)
        return image, entry, format_data(self.image_directory_path, entry, self.system_message)


def create_dataset(dataset_path, system_message) -> Tuple:
    train_dataset = JSONLDataset(
        jsonl_file_path=f"{dataset_path}/train/annotations.jsonl",
        image_directory_path=f"{dataset_path}/train",
        system_message=system_message
    )

    valid_dataset = JSONLDataset(
        jsonl_file_path=f"{dataset_path}/valid/annotations.jsonl",
        image_directory_path=f"{dataset_path}/valid",
        system_message=system_message
    )

    test_dataset = JSONLDataset(
        jsonl_file_path=f"{dataset_path}/test/annotations.jsonl",
        image_directory_path=f"{dataset_path}/test",
        system_message=system_message
    )
    return train_dataset, valid_dataset, test_dataset


<<<<<<< HEAD
def download_dataset_rob():
    dataset = download_dataset("https://app.roboflow.com/roboflow-jvuqo/pallet-load-manifest-json/2", "jsonl")
    # head_5 = f"!head -n 5 {dataset.location}/train/annotations.jsonl"
    # os.system(head_5)
=======
def download_dataset_from_roboflow(dataset_path):
    download_dataset(
        dataset_url="https://app.roboflow.com/roboflow-jvuqo/pallet-load-manifest-json/2",
        location=dataset_path,
        model_format="jsonl"
    )

    train_annotations = os.path.join(
        dataset_path, 'train', 'annotations.jsonl')
    valid_annotations = os.path.join(
        dataset_path, 'valid', 'annotations.jsonl')
    test_annotations = os.path.join(dataset_path, 'test', 'annotations.jsonl')

>>>>>>> 7101828b6d296345f6886969668d4a23bbe959db
    print("Dataset downloaded successfully!")

    # Add prompt to all splits
    def add_prompt_to_file(file_path):
        cmd = f"sed -i 's/<JSON>/extract data in JSON format/g' {file_path}"
        subprocess.run(cmd, shell=True)

    add_prompt_to_file(train_annotations)
    add_prompt_to_file(valid_annotations)
    add_prompt_to_file(test_annotations)

    subprocess.run(f"head -n 5 {train_annotations}", shell=True)

    print("Prompt added successfully!")


if __name__ == "__main__":
<<<<<<< HEAD
    download_dataset_rob()
    
    # config = {
    #     "dataset_path": "data",
    #     "dataloader": {
    #         "batch_size": 16,
    #         "num_workers": 10,
    #         "shuffle": True
    #     }
    # }
    # train_loader, valid_loader, test_loader = create_dataset(config)
    # print(train_loader)
    # print(valid_loader)
    # print(test_loader)
    # print("Dataset created successfully!")
=======
    dataset_path = "./pallet-load-manifest-json-2"
    if not os.path.exists(dataset_path):
        print("Downloading Dataset")
        download_dataset_from_roboflow(dataset_path)

    config = {
        "dataset_path": "pallet-load-manifest-json-2",
    }
    train_loader, valid_loader, test_loader = create_dataset(dataset_path)
    print("len(train_loader)", len(train_loader))
    print("len(valid_loader)", len(valid_loader))
    print("len(test_loader)", len(test_loader))
    print("Dataset created successfully!")
>>>>>>> 7101828b6d296345f6886969668d4a23bbe959db
