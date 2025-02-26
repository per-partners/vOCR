import gc
import os
import torch
import yaml
import lightning as L

# Load custom modules
from src.dataset import create_dataset
from src.model import init_model
from src.trainer import Qwen2_5_Trainer
from src.utilities import SaveCheckpoint, load_config, early_stopping_callback


def main():
    # Load configuration
    config = load_config("config.yaml")
    train_dataset, val_dataset, test_dataset = create_dataset(
        data_dir=config["data_dir"])
    model, processor = init_model(config)
    model_module = Qwen2_5_Trainer(config, processor, model)
    trainer = L.Trainer(
        accelerator=config["training_hp"]["accelerator"],
        devices=config["training_hp"],
        max_epochs=config["training_hp"]["max_epochs"],
        accumulate_grad_batches=config["training_hp"]["accumulate_grad_batches"],
        check_val_every_n_epoch=config["training_hp"]["check_val_every_n_epoch"],
        gradient_clip_val=config["training_hp"]["gradient_clip_val"],
        limit_val_batches=config["training_hp"]["limit_val_batches"],
        num_sanity_val_steps=config["training_hp"]["num_sanity_val_steps"],
        log_every_n_steps=config["training_hp"]["log_every_n_steps"],
        num_nodes=config["training_hp"]["num_nodes"],
        strategy=config["training_hp"]["strategy"],
        callbacks=[SaveCheckpoint(
            result_path=config["result_path"]), early_stopping_callback]
    )

    mode = config["mode"]
    if mode == "train":
        trainer.fit(model_module)
    elif mode == "resume":
        pass
    elif mode == "eval":
        pass
    elif mode == "predict":
        pass
    else:
        raise ValueError(
            "Invalid mode. Please select train, resume, eval or predict.")

    # Manual cleanup after each epoch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
