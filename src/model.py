import torch

import lightning as L
from nltk import edit_distance
from torch.optim import AdamW
from utilities import train_collate_fn
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig
from transformers import BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


def init_model(config: dict):
    print("Initializing model")

    # Use QLORA if use_qlora is True
    use_qlora = config["model_hp"].get("use_qlora", False)
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16
        )

    # Construct Qwen model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_hp"]["model_id"],
        device_map=config["training_hp"]["device"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if config["training_hp"]["bf16"] else torch.float32)

    # Use PEFT if use_lora is True
    use_lora = config["model_hp"].get("use_lora", False)
    lora_config = None
    if use_lora:
        # Define the LoraConfig
        lora_config = LoraConfig(
            lora_alpha=config["lora_settings"]["lora_alpha"],
            lora_dropout=config["lora_settings"]["lora_dropout"],
            # rank of the low-rank approximation
            r=config["lora_settings"]["rank"],
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        # Wrap the model with PEFT
        model = get_peft_model(model, lora_config)

    print("Initializing processor")
    # Construct Qwen processor
    processor = Qwen2_5_VLProcessor.from_pretrained(
        config["model_hp"]["model_id"], min_pixels=config["model_hp"]["min_pixels"] * 28 * 28, max_pixels=config["model_hp"]["max_pixels"] * 28 * 28)

    # print("=" * 50)
    # model.print_trainable_parameters()
    # print("=" * 50)
    return model, processor


if __name__ == "__main__":
    config = {
        "model_hp": {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "min_pixels": 256,
            "max_pixels": 1280
        },

        "training_hp": {
            "device": "cuda",
            "bf16": True
        }
    }
    model, processor = init_model(config)
    print(model)
    print(processor)
