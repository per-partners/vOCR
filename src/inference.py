import torch
from src.utilities import process_vision_info, side_by_side_diff_divs
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from difflib import SequenceMatcher
from IPython.core.display import display, HTML
# from IPython import display




def test(config: dict):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["test"]["test_weight_path"],
        device_map="auto",
        torch_dtype=torch.bfloat16)
    processor = Qwen2_5_VLProcessor.from_pretrained(
        "/content/qwen2.5-3b-instruct-palette-manifest/latest",
        min_pixels=config["model_hp"]["min_pixels"],
        max_pixels=config["model_hp"]["max_pixels"])
    return model, processor


def run_inference(config: dict):
    model, processor = test(config)
    text = processor.apply_chat_template(
        config["test"]["label"], tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(config["test"]["label"])

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    )
    inputs = inputs.to(config["train"]["device"])

    generated_ids = model.generate(
        **inputs, max_new_tokens=config["test"]["max_new_tokens"])
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids
        in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]


if __name__=="__main__":
    config = {
        "test": {
            "test_weight_path": "/content/qwen2.5-3b-instruct-palette-manifest/latest",
            "label": "A person",
            "max_new_tokens": 1024
        },
        "model_hp": {
            "min_pixels": 1,
            "max_pixels": 3
        },
        "train": {
            "device": "cuda"
        }
    }
    suffix = "A person with a hat"
    generated_suffix = run_inference(config)
    html_diff = side_by_side_diff_divs(suffix, generated_suffix)
    display(HTML(html_diff))