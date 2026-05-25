export CUDA_VISIBLE_DEVICES=3   # train.sh와 같은 GPU 쓰려면
# 학습된 LoRA 적용
# uv run python inference.py --verify
# 베이스(스크래치)만 — 어댑터는 무시하고 동일 train_set으로 비교
uv run python inference.py --verify --base-only