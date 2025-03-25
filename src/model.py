import lightning as L
import torch
from nltk import edit_distance
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
from torch.optim import AdamW
import os
from transformers import (BitsAndBytesConfig,
                          Qwen2_5_VLForConditionalGeneration,
                          Qwen2_5_VLProcessor, AutoProcessor, Gemma3ForConditionalGeneration)

from src.utilities import train_collate_fn

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")


def init_model_gemma(config: dict):
    print("Initializing model")
    # Use QLORA if use_qlora is True
    use_qlora = config["model_hp"].get("use_qlora", False)
    bnb_config = None
    if use_qlora:
        print("QLORA Activated")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16
        )

    # Construct Gemma model
    model = Gemma3ForConditionalGeneration.from_pretrained(
        config["model_hp"]["model_id"],
        quantization_config=bnb_config,
        token=hf_token,
        torch_dtype=torch.bfloat16 if config["training_hp"]["bf16"] else torch.float32
    ).eval()

    # Use PEFT if use_lora is True
    use_lora = config["model_hp"].get("use_lora", False)
    if use_lora:
        print("LORA Activated")
        lora_config = LoraConfig(
            lora_alpha=config["lora_settings"]["lora_alpha"],
            lora_dropout=config["lora_settings"]["lora_dropout"],
            r=config["lora_settings"]["rank"],
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    print("Initializing processor")
    processor = AutoProcessor.from_pretrained(
        config["model_hp"]["model_id"], token=hf_token)

    return model, processor


def init_model(config: dict):
    print("Initializing model")

    # Use QLORA if use_qlora is True
    use_qlora = config["model_hp"].get("use_qlora", False)
    bnb_config = None
    if use_qlora:
        print("QLORA Activated")
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
        token=hf_token,
        torch_dtype=torch.bfloat16 if config["training_hp"]["bf16"] else torch.float32)

    # Use PEFT if use_lora is True
    use_lora = config["model_hp"].get("use_lora", False)
    lora_config = None
    if use_lora:
        print("LORA Activated")
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
    print("Min Pixels:", config["model_hp"].get("min_pixels"))

    print("Max Pixels:", config["model_hp"].get("max_pixels"))
    processor = Qwen2_5_VLProcessor.from_pretrained(
        config["model_hp"]["model_id"],
        use_auth_token=hf_token, min_pixels=config["model_hp"]["min_pixels"] * 28 * 28, max_pixels=config["model_hp"]["max_pixels"] * 28 * 28)

    print('here')

    return model, processor


if __name__ == "__main__":
    config = {
        "model_hp": {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "min_pixels": 256,
            "max_pixels": 1280,
            "use_lora": True,
            "use_qlora": True,
        },
        "lora_settings": {
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "rank": 8,
        },
        "training_hp": {
            "batch_size": 8,
            "bf16": True,
            "device": "cuda",
            "max_epochs": 10,
            "lr": 2e-4,
            "check_val_every_n_epoch": 2,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 8,
            "num_nodes": 1,
            "warmup_steps": 50,
            "devices": 0,
            "accelerator": "gpu",
            "strategy": "ddp",
            "result_path": "qwen2.5-3b-instruct",
        },
        "dataloader": {
            "batch_size": 8,
            "num_workers": 4,
            "shuffle": True,
            "pin_memory": True,
        },
        "model": "train",  # possible values: test, resume, train
        "system_message": (
            "You are a Vision Language Model specialized in extracting structured data from visual representations of palette manifests. "
            "Your task is to analyze the provided image of a palette manifest and extract the relevant information into a well-structured JSON format. "
            "The palette manifest includes details such as item names, quantities, dimensions, weights, and other attributes. "
            "Focus on identifying key data fields and ensuring the output adheres to the requested JSON structure. "
            "Provide only the JSON output based on the extracted information. Avoid additional explanations or comments."
        ),
        "dataset_path": "assets/dataset",
        "test": {
            "test_weight_path": None,
            "max_new_tokens": 1024,
            "label": None,
        },
    }

    model, processor = init_model(config)
    print(model)
    print(processor)
