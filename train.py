#!/usr/bin/env python3
"""LoRA SFT on RAG-KV style chat JSON (assistant tokens only contribute to loss)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)


def _as_token_id_list(ids: Any) -> List[int]:
    """Normalize apply_chat_template(tokenize=True) output across transformers versions."""
    if ids is None:
        raise TypeError("apply_chat_template returned None")
    # Newer HF returns BatchEncoding with input_ids (possibly batched).
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if isinstance(ids, torch.Tensor):
        t = ids.detach().cpu()
        if t.dim() == 2 and t.shape[0] == 1:
            t = t[0]
        return [int(x) for x in t.flatten().tolist()]
    try:
        import numpy as np

        if isinstance(ids, np.ndarray):
            a = ids
            if a.ndim == 2 and a.shape[0] == 1:
                a = a[0]
            return [int(x) for x in a.astype(np.int64).flatten().tolist()]
    except ImportError:
        pass
    if isinstance(ids, (list, tuple)):
        # Unwrap batch-of-one: [[1,2,3]] -> [1,2,3]
        if (
            len(ids) == 1
            and isinstance(ids[0], (list, tuple))
            and (not ids[0] or isinstance(ids[0][0], int))
        ):
            ids = ids[0]
        return [int(x) for x in ids]
    raise TypeError(f"Unexpected apply_chat_template tokenize output type: {type(ids)!r}")


def _chat_template_kwargs() -> Dict[str, Any]:
    """Match vLLM `chat_template_kwargs.enable_thinking: false` when supported."""
    return {"enable_thinking": False}


def apply_chat_template_ids(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> List[int]:
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
    )
    extra = _chat_template_kwargs()
    try:
        out = tokenizer.apply_chat_template(messages, **kwargs, **extra)
    except TypeError:
        out = tokenizer.apply_chat_template(messages, **kwargs)
    return _as_token_id_list(out)


@dataclass
class CausalLMDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_jsonl_messages(path: Path) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not msgs or msgs[-1]["role"] != "assistant":
                raise ValueError(f"Each line must end with assistant message: {obj.get('id')!r}")
            rows.append({"messages": msgs, "id": obj.get("id", "")})
    return Dataset.from_list(rows)


def tokenize_batch(
    examples: Dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> Dict[str, Any]:
    input_ids_out: List[List[int]] = []
    labels_out: List[List[int]] = []
    for messages in examples["messages"]:
        if not isinstance(messages, list):
            messages = list(messages)
        prompt_messages = messages[:-1]
        assistant_message = messages[-1]
        if assistant_message["role"] != "assistant":
            raise ValueError("Last message must be assistant")

        prompt_ids = apply_chat_template_ids(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        full_ids = apply_chat_template_ids(
            tokenizer,
            messages,
            add_generation_prompt=False,
        )
        if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
            LOGGER.warning(
                "Prompt token prefix mismatch; training loss on full sequence for this row. "
                "prompt_len=%s full_len=%s",
                len(prompt_ids),
                len(full_ids),
            )
            cut = 0
        else:
            cut = len(prompt_ids)
        labels = [-100] * cut + full_ids[cut:]
        input_ids = full_ids
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
        input_ids_out.append(input_ids)
        labels_out.append(labels)

    return {"input_ids": input_ids_out, "labels": labels_out}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data1/share/maruchanpark/models/rag-kv/Qwen3.5-9B"),
        help="Local HF checkpoint (same tree as docker-compose host mount).",
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "train_set" / "train.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "lora_adapter",
    )
    p.add_argument("--epochs", type=float, default=15.0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module suffixes for LoRA.",
    )
    p.add_argument("--logging-steps", type=int, default=1)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    ds = load_jsonl_messages(args.train_jsonl)
    remove_cols = ds.column_names
    tokenized = ds.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=remove_cols,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=CausalLMDataCollator(pad_token_id=int(tokenizer.pad_token_id)),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    LOGGER.info("Saved adapter and tokenizer to %s", args.output_dir)


if __name__ == "__main__":
    main()
