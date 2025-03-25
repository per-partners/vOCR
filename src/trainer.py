import lightning as L
import torch
from nltk import edit_distance
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import create_dataset
from src.model import init_model
from src.utilities import validation_collate_fn, train_collate_fn


class Qwen2_5_Trainer(L.LightningModule):
    def __init__(self, config, model, processor):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataset, self.valid_dataset, self.test_dataset = create_dataset(
            dataset_path=config["dataset_path"], system_message=config["system_message"])

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, image_grid_thw, suffixes = batch
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=1024
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids
            in zip(input_ids, generated_ids)]

        generated_suffixes = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        scores = []
        for generated_suffix, suffix in zip(generated_suffixes, suffixes):
            # score = edit_distance(generated_suffix, suffix)
            score = 100
            score = score / max(len(generated_suffix), len(suffix))
            scores.append(score)
            
            print("generated_suffix", generated_suffix)
            print("suffix", suffix)
            print("score", score)

        score = sum(scores) / len(scores)
        self.log("val_edit_distance", score, prog_bar=True,
                 logger=True, batch_size=self.config["dataloader"]["batch_size"])
        return scores
    
    
    def test_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, image_grid_thw, suffixes = batch
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=1024
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids
            in zip(input_ids, generated_ids)]

        generated_suffixes = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        scores = []
        for generated_suffix, suffix in zip(generated_suffixes, suffixes):
            score = edit_distance(generated_suffix, suffix)
            score = score / max(len(generated_suffix), len(suffix))
            scores.append(score)
            print("generated_suffix", generated_suffix)
            print("suffix", suffix)
            print("score", score)

        score = sum(scores) / len(scores)
        self.log("test_edit_distance", score, prog_bar=True,
                 logger=True, batch_size=self.config["train_hp"]["batch_size"])
        return scores
    

    def configure_optimizers(self):
        return AdamW(self.model.parameters(),
                     lr=self.config["training_hp"]["lr"])

    # Ignore the loading from these part of the code
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config["dataloader"]["batch_size"],
                          collate_fn=train_collate_fn,
                          num_workers=self.config["dataloader"]["num_workers"],
                          shuffle=self.config["dataloader"]["shuffle"])

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.config["dataloader"]["batch_size"],
                          collate_fn=validation_collate_fn,
                          num_workers=self.config["dataloader"]["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.config["dataloader"]["batch_size"],
                          collate_fn=validation_collate_fn,
                          num_workers=self.config["dataloader"]["num_workers"])


if __name__ == "__main__":
    config = {
        "data_dir": "data",
        "dataloader": {
            "batch_size": 16,
            "num_workers": 10,
            "shuffle": True},

        "training_hp": {
            "lr": 1e-5,
            "batch_size": 16},

        "model_hp": {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "min_pixels": 1,
            "max_pixels": 1,
            "use_qlora": False,
            "use_lora": False,
            "device": "cuda",
            "bf16": True,
        }
    }
    model, processor = init_model(config)
    module = Qwen2_5_Trainer(config, model, processor)
