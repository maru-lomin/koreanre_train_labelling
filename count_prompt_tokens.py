#!/usr/bin/env python3
"""Count chat-template prompt / full / assistant token lengths for train JSONL rows."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

from train import apply_chat_template_ids


def _percentile(sorted_vals: List[int], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    k = (n - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _row_lengths(
    tokenizer: Any,
    messages: List[Dict[str, str]],
) -> Tuple[int, int, int, bool]:
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Each row must have messages ending with role=assistant")

    prompt_messages = messages[:-1]
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
    mismatch = len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids
    cut = 0 if mismatch else len(prompt_ids)
    assistant_len = len(full_ids) - cut
    return len(prompt_ids), len(full_ids), assistant_len, mismatch


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            msgs = obj.get("messages")
            if not msgs:
                raise ValueError(f"{path}:{line_no}: missing messages (id={obj.get('id')!r})")
            rows.append({"id": obj.get("id", ""), "messages": msgs, "_source": str(path), "_line": line_no})
    return rows


def _summarize(name: str, values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"name": name, "count": 0}
    s = sorted(values)
    return {
        "name": name,
        "count": len(values),
        "min": s[0],
        "max": s[-1],
        "mean": round(statistics.mean(values), 2),
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        "p50": round(_percentile(s, 50)),
        "p90": round(_percentile(s, 90)),
        "p95": round(_percentile(s, 95)),
        "p99": round(_percentile(s, 99)),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--jsonl",
        type=Path,
        nargs="+",
        default=[root / "train_set" / "train.jsonl"],
        help="One or more JSONL files (same format as train.py: messages, optional id).",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data1/share/maruchanpark/models/rag-kv/Qwen3.5-9B"),
        help="HF tokenizer path (must match training).",
    )
    p.add_argument(
        "--per-row",
        action="store_true",
        help="Print one TSV line per row: source, line, id, prompt_tokens, full_tokens, assistant_tokens, prefix_mismatch",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_out",
        help="Emit summary as JSON on stdout (per-row still TSV if combined).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p).resolve() for p in args.jsonl]
    for path in paths:
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)

    prompt_lens: List[int] = []
    full_lens: List[int] = []
    asst_lens: List[int] = []
    mismatches = 0
    per_row: List[Dict[str, Any]] = []

    for path in paths:
        for row in _load_rows(path):
            try:
                pt, ft, at, bad = _row_lengths(tokenizer, row["messages"])
            except Exception as e:
                raise RuntimeError(
                    f"{row['_source']}:{row['_line']}: tokenization failed (id={row['id']!r}): {e}"
                ) from e
            prompt_lens.append(pt)
            full_lens.append(ft)
            asst_lens.append(at)
            if bad:
                mismatches += 1
            per_row.append(
                {
                    "source": row["_source"],
                    "line": row["_line"],
                    "id": row["id"],
                    "prompt_tokens": pt,
                    "full_tokens": ft,
                    "assistant_tokens": at,
                    "prefix_mismatch": bad,
                }
            )

    if args.per_row:
        print("source\tline\tid\tprompt_tokens\tfull_tokens\tassistant_tokens\tprefix_mismatch")
        for r in per_row:
            print(
                f"{r['source']}\t{r['line']}\t{r['id']}\t{r['prompt_tokens']}\t"
                f"{r['full_tokens']}\t{r['assistant_tokens']}\t{r['prefix_mismatch']}"
            )

    summary = {
        "model_path": str(args.model_path),
        "files": [str(p) for p in paths],
        "rows": len(per_row),
        "prefix_mismatch_rows": mismatches,
        "prompt_tokens": _summarize("prompt_tokens", prompt_lens),
        "full_tokens": _summarize("full_tokens", full_lens),
        "assistant_tokens": _summarize("assistant_tokens", asst_lens),
    }

    if args.json_out:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif not args.per_row:
        print(f"model_path: {summary['model_path']}")
        print(f"files: {', '.join(summary['files'])}")
        print(f"rows: {summary['rows']}")
        print(f"prefix_mismatch_rows: {summary['prefix_mismatch_rows']}")
        for key in ("prompt_tokens", "full_tokens", "assistant_tokens"):
            block = summary[key]
            print(f"\n[{key}]")
            if block["count"] == 0:
                print("  (no rows)")
                continue
            print(
                f"  count={block['count']} min={block['min']} max={block['max']} "
                f"mean={block['mean']} stdev={block['stdev']}"
            )
            print(
                f"  p50={block['p50']} p90={block['p90']} p95={block['p95']} p99={block['p99']}"
            )
    else:
        # --per-row only: keep stdout as TSV; note counts on stderr
        print(
            f"# rows={len(per_row)} prefix_mismatch={mismatches}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
