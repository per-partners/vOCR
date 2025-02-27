from IPython.core.display import display, HTML
from difflib import SequenceMatcher
from qwen_vl_utils import process_vision_info
import yaml
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os

# Callbacks
early_stopping_callback = EarlyStopping(
    monitor="val_edit_distance", patience=3, verbose=False, mode="min")


def format_data(image_directory_path, entry, system_message):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_directory_path + "/" + entry["image"],
                },
                {
                    "type": "text",
                    "text": entry["prefix"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": entry["suffix"]}],
        },
    ]


def train_collate_fn(batch, processor):
    _, _, examples = zip(*batch)

    texts = [
        processor.apply_chat_template(example, tokenize=False)
        for example
        in examples
    ]
    image_inputs = [
        process_vision_info(example)[0]
        for example
        in examples
    ]

    model_inputs = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True
    )

    labels = model_inputs["input_ids"].clone()

    # mask system message and image token IDs in the labels
    for i, example in enumerate(examples):
        sysuser_conv = example[:-1]
        sysuser_text = processor.apply_chat_template(
            sysuser_conv, tokenize=False)
        sysuser_img, _ = process_vision_info(sysuser_conv)

        sysuser_inputs = processor(
            text=[sysuser_text],
            images=[sysuser_img],
            return_tensors="pt",
            padding=True,
        )

        sysuser_len = sysuser_inputs["input_ids"].shape[1]
        labels[i, :sysuser_len] = -100

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs["image_grid_thw"]

    return input_ids, attention_mask, pixel_values, image_grid_thw, labels


def evaluation_collate_fn(batch, processor):
    _, data, examples = zip(*batch)
    suffixes = [d["suffix"] for d in data]

    # drop the assistant portion so the model must generate it
    examples = [e[:2] for e in examples]

    texts = [
        processor.apply_chat_template(example, tokenize=False)
        for example
        in examples
    ]
    image_inputs = [
        process_vision_info(example)[0]
        for example
        in examples
    ]

    model_inputs = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True)

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs["image_grid_thw"]
    return input_ids, attention_mask, pixel_values, image_grid_thw, suffixes


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class SaveCheckpoint(Callback):
    def __init__(self, result_path):
        self.result_path = result_path
        self.epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        checkpoint_path = f"{self.result_path}/{self.epoch}"
        os.makedirs(checkpoint_path, exist_ok=True)

        pl_module.processor.save_pretrained(checkpoint_path)
        pl_module.model.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        self.epoch += 1

    def on_train_end(self, trainer, pl_module):
        checkpoint_path = f"{self.result_path}/latest"
        os.makedirs(checkpoint_path, exist_ok=True)
        pl_module.processor.save_pretrained(checkpoint_path)
        pl_module.model.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


def side_by_side_diff_divs(text1, text2):
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    original_output = []
    modified_output = []

    for line1, line2 in zip(lines1, lines2):
        words1 = line1.split()
        words2 = line2.split()

        matcher = SequenceMatcher(None, words1, words2)

        original_line = []
        modified_line = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                original_line.append(
                    f"<span class='diff-remove'>{' '.join(words1[i1:i2])}</span>")
                modified_line.append(
                    f"<span class='diff-add'>{' '.join(words2[j1:j2])}</span>")
            elif tag == 'delete':
                original_line.append(
                    f"<span class='diff-remove'>{' '.join(words1[i1:i2])}</span>")
            elif tag == 'insert':
                modified_line.append(
                    f"<span class='diff-add'>{' '.join(words2[j1:j2])}</span>")
            elif tag == 'equal':
                original_line.append(' '.join(words1[i1:i2]))
                modified_line.append(' '.join(words2[j1:j2]))

        original_output.append(' '.join(original_line) + "<br>")
        modified_output.append(' '.join(modified_line) + "<br>")

    original_html = "<br>" + ''.join(original_output) + "<br>"
    modified_html = "<br>" + ''.join(modified_output) + "<br>"

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
            .container {{ display: flex; align-items: flex-start; }}
            .column {{
                flex: 1;
                padding: 10px;
                white-space: pre-wrap;
                text-align: left;
            }}
            .diff-remove {{
                background-color: #d9534f;  /* Dark red */
                color: white;
                text-decoration: line-through;
                border-radius: 4px;
                padding: 2px 4px;
            }}
            .diff-add {{
                background-color: #5cb85c;  /* Dark green */
                color: white;
                border-radius: 4px;
                padding: 2px 4px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="column" style="border-right: 1px solid #ccc;">
                {original_html}
            </div>
            <div class="column">
                {modified_html}
            </div>
        </div>
    </body>
    </html>
    """
    return html
