export CUDA_VISIBLE_DEVICES=3
uv run python train.py --epochs 15 --output-dir ./outputs/lora_adapter
