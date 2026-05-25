#!/usr/bin/env python3
"""Load base (+ optional LoRA) and generate JSON (for memorization checks on train_set)."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def _chat_template_kwargs() -> Dict[str, Any]:
    return {"enable_thinking": False}


def apply_chat_template_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    extra = _chat_template_kwargs()
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, **extra)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data1/share/maruchanpark/models/rag-kv/Qwen3.5-9B"),
    )
    p.add_argument(
        "--adapter-path",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "lora_adapter",
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "train_set" / "train.jsonl",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--verify",
        action="store_true",
        help="Parse expected vs generated JSON and print match rate.",
    )
    p.add_argument(
        "--base-only",
        action="store_true",
        help="Do not load LoRA even if --adapter-path exists (compare scratch/base weights).",
    )
    return p.parse_args()


def load_examples(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.base_only:
        model = base
        LOGGER.info("Base-only inference (--base-only); LoRA not loaded.")
    elif args.adapter_path.exists() and any(args.adapter_path.iterdir()):
        model = PeftModel.from_pretrained(base, str(args.adapter_path))
        LOGGER.info("Loaded LoRA from %s", args.adapter_path)
    else:
        model = base
        LOGGER.warning("Adapter path missing or empty; running base model only.")

    model.eval()

    examples = load_examples(args.train_jsonl)
    ok = 0
    for ex in examples:
        eid = ex.get("id", "")
        messages: List[Dict[str, str]] = ex["messages"]
        expected = messages[-1]["content"]
        prompt_messages = messages[:-1]
        prompt = apply_chat_template_prompt(tokenizer, prompt_messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion = strip_json_fence(completion)

        print(f"\n=== id={eid} ===")
        print("--- expected ---")
        print(expected.strip())
        print("--- generated ---")
        print(completion)

        if args.verify:
            try:
                exp_obj = json.loads(expected.strip())
                gen_obj = json.loads(completion)
                if exp_obj == gen_obj:
                    ok += 1
                    print("json_match: True")
                else:
                    print("json_match: False (object differ)")
            except json.JSONDecodeError as err:
                print(f"json_match: False (parse error: {err})")

    if args.verify:
        print(f"\nverify_summary: {ok}/{len(examples)} exact JSON matches")


if __name__ == "__main__":
    main()
