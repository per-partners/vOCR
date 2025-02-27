from roboflow import download_dataset
import json
import os
from typing import Tuple

from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from src.utilities import format_data, evaluation_collate_fn, train_collate_fn
from torch.utils.data import DataLoader


class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

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
        return image, entry, format_data(self.image_directory_path, entry)


def create_dataset(config: dict) -> Tuple:
    train_dataset = JSONLDataset(
        jsonl_file_path=f"{config["dataset_path"]}/train/annotations.jsonl",
        image_directory_path=f"{config["dataset_path"]}/train")

    valid_dataset = JSONLDataset(
        jsonl_file_path=f"{config["dataset_path"]}/valid/annotations.jsonl",
        image_directory_path=f"{config["dataset_path"]}/valid")

    test_dataset = JSONLDataset(
        jsonl_file_path=f"{config["dataset_path"]}/test/annotations.jsonl",
        image_directory_path=f"{config["dataset_path"]}/test",
    )
    return train_dataset, valid_dataset, test_dataset


def download_dataset():
    dataset = download_dataset("https://app.roboflow.com/roboflow-jvuqo/pallet-load-manifest-json/2", "jsonl")
    head_5 = f"!head -n 5 {dataset.location}/train/annotations.jsonl"
    os.system(head_5)
    print("Dataset downloaded successfully!")
    add_prompt = f"!sed -i 's/<JSON>/extract data in JSON format/g' {dataset.location}/train/annotations.jsonl
                   !sed -i 's/<JSON>/extract data in JSON format/g' {dataset.location}/valid/annotations.jsonl
                   !sed -i 's/<JSON>/extract data in JSON format/g' {dataset.location}/test/annotations.jsonl"
    os.system(add_prompt)
    print("Prompt added successfully!")
    
    
if __name__ == "__main__":
    download_dataset()
    
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
