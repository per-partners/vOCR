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
    train_dataset, val_dataset, test_dataset = create_dataset(data_dir=config["data_dir"])
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
    callbacks=[SaveCheckpoint(result_path=config["result_path"]), early_stopping_callback]
)

    trainer.fit(model_module)
    mode = config["mode"]
    if mode == "train":
        model = trainer.train(train_dataset, val_dataset)
    
    elif mode == "resume":
        model = trainer.resume(train_dataset, val_dataset)
    
    elif mode == "eval":
        metrics = trainer.evaluate(val_dataset)
        print("=" * 50)
        print("=" * 40)
        print(f"Metrics: {metrics}")
        print("=" * 40)
        print("=" * 50)
    elif mode == "predict":
        test_image = config["test_image"]
        output_dir = config["output_dir"]
        predictions_dir = os.path.join(output_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        prediction_text = trainer.predict(test_image)
        # save_prediction(prediction_text, test_image, output_dir)
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
