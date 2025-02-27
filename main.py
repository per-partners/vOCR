import gc

import lightning as L
import torch

# Load custom modules
# from src.dataset import create_dataset
from src.model import init_model
from src.trainer import Qwen2_5_Trainer
from src.utilities import SaveCheckpoint, early_stopping_callback, load_config
from src import globals_manager

def main():
    # Load configuration
    config = load_config("config.yaml")
    model, processor = init_model(config)
    model_module = Qwen2_5_Trainer(config, model, processor)
    globals_manager.processor = processor
    training_config = config["training_hp"]
    trainer = L.Trainer(
        accelerator=training_config["accelerator"],
        devices=training_config["devices"],
        max_epochs=training_config["max_epochs"],
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        check_val_every_n_epoch=training_config["check_val_every_n_epoch"],
        gradient_clip_val=training_config["gradient_clip_val"],
        limit_val_batches=training_config["limit_val_batches"],
        num_sanity_val_steps=training_config["num_sanity_val_steps"],
        log_every_n_steps=training_config["log_every_n_steps"],
        num_nodes=training_config["num_nodes"],
        strategy=training_config["strategy"],
        callbacks=[SaveCheckpoint(
            result_path=training_config["result_path"]), early_stopping_callback]
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
