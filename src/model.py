import torch
from transformers import BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


def init_model(config: dict):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_hp"]["model_id"],
        device_map=config["Training_HyperParameter"]["DEVICE"],
        torch_dtype=torch.bfloat16 if config["Training_HyperParameter"]["bf16"] else torch.float32)

    processor = Qwen2_5_VLProcessor.from_pretrained(
        config["model_hp"]["model_id"], min_pixels=config["model_hp"]["MIN_PIX"] * 28 * 28, max_pixels=config["model_hp"]["MAX_PIX"] * 28 * 28)

    print("=" * 50)
    model.print_trainable_parameters()
    print("=" * 50)
    return model, processor


if __name__ == "__main__":
    config = {
        "model_hp": {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "MIN_PIX": 256,
            "MAX_PIX": 1280
        },
        
        "Training_HyperParameter": {
            "DEVICE": "cuda",
            "bf16": True
        }
    }
    model, processor = init_model(config)
    print(model)
    print(processor)
