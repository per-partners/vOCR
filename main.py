import gc
import os

import torch

from src.dataset import create_data
from src.trainer import MultiLangTrainer
from src.utilities import init_model, load_config, save_prediction


# Main Function
def main():
    # Load configuration
    config = load_config("config.yaml")

    model, processor = init_model(config)
    train_dataset, val_dataset = create_data(
        data_dir=config["data_dir"], processor=processor, config=config
    )

    trainer = MultiLangTrainer(config, model, processor)
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
        save_prediction(prediction_text, test_image, output_dir)
    else:
        raise ValueError("Invalid mode. Please select train, resume, eval or predict.")

    # Manual cleanup after each epoch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
