# Single GPU (set one device index)
# export CUDA_VISIBLE_DEVICES=3
# uv run python train.py --epochs 15 --output-dir ./outputs/lora_adapter

# uv run python train.py --epochs 15 --output-dir ./train_1/lora_adapter \
#     --train-jsonl ./train_set/train_data_1.jsonl --max-seq-length 3072 \
#     --per-device-batch-size 4

# Multi-GPU DDP (4 processes = 4 GPUs). Global batch ≈ per_device * grad_accum * 4.
# unset CUDA_VISIBLE_DEVICES  # optional: use default 0,1,2,...
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 uv run torchrun --standalone --nproc_per_node=6 train.py \
    --epochs 15 --output-dir ./train_3/lora_adapter \
    --train-jsonl ./train_set/train_data_3.jsonl --max-seq-length 3072 \
    --per-device-batch-size 1
