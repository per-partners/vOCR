# Install the packages in r1-v .
cd src/setup
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation
pip install lightning

# fix transformers version
pip install -q "transformers>=4.49.0" accelerate peft bitsandbytes "qwen-vl-utils[decord]==0.0.8"