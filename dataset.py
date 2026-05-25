#!/usr/bin/env python3
"""
Convert RAG-KV demo outputs (llm_request.json + llm_response.json) into train_set-style JSONL.

Each line: {"id": "...", "messages": [system, user, assistant]}

Training rows omit ``confidence``: the user prompt's "### Output JSON schema" block is rewritten
without the confidence field, and the assistant JSON is re-serialized without that key (so
labels match the prompt).

--src may be:
  - A single run directory that contains llm_request.json and llm_response.json, or
  - A parent directory (e.g. outputs/rag_kv_demo) whose immediate child dirs contain those files.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger(__name__)

# User prompt template (RAG-KV): remove confidence from the schema snippet so it matches labels.
_RE_USER_SCHEMA_CONFIDENCE_ML = re.compile(r",\s*\n\s*\"confidence\"\s*:\s*<float>", re.MULTILINE)
_RE_USER_SCHEMA_CONFIDENCE_1L = re.compile(r",\s*\"confidence\"\s*:\s*<float>")
_RE_USER_SCHEMA_CONFIDENCE_FIRST = re.compile(r"\n\s*\"confidence\"\s*:\s*<float>\s*,", re.MULTILINE)

# Assistant JSON fallback when json.loads fails (e.g. trailing junk).
_RE_ASST_CONF_ML = re.compile(
    r",\s*\n\s*\"confidence\"\s*:\s*(-?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?|null|true|false)\s*",
    re.MULTILINE,
)
_RE_ASST_CONF_1L = re.compile(
    r",\s*\"confidence\"\s*:\s*(-?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?|null|true|false)\s*"
)
_RE_ASST_CONF_FIRST = re.compile(
    r"\n\s*\"confidence\"\s*:\s*(-?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?|null|true|false)\s*,",
    re.MULTILINE,
)


def strip_confidence_from_user_prompt(text: str) -> str:
    """Drop the confidence field line from the Output JSON schema in the user message."""
    out = _RE_USER_SCHEMA_CONFIDENCE_ML.sub("", text)
    out = _RE_USER_SCHEMA_CONFIDENCE_1L.sub("", out)
    out = _RE_USER_SCHEMA_CONFIDENCE_FIRST.sub("\n", out)
    return out


def strip_confidence_from_assistant(content: str) -> str:
    """Parse assistant JSON, remove ``confidence``, re-serialize (preserve pretty vs one-line)."""
    raw = content.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()

    use_pretty = "\n" in content
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        fixed = _RE_ASST_CONF_ML.sub("", raw)
        fixed = _RE_ASST_CONF_1L.sub("", fixed)
        fixed = _RE_ASST_CONF_FIRST.sub("\n", fixed)
        if fixed != raw:
            LOGGER.warning("Assistant content was not valid JSON; stripped confidence via regex.")
            return fixed.strip()
        LOGGER.warning("Assistant JSON parse failed; leaving content unchanged.")
        return content

    if not isinstance(obj, dict):
        return content
    obj.pop("confidence", None)

    if use_pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Run folder with llm_*.json, or parent of run folders (e.g. outputs/rag_kv_demo).",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("./dataset"),
        help="Output directory (will create train.jsonl here by default).",
    )
    p.add_argument(
        "--output-name",
        default="train.jsonl",
        help="JSONL filename under --dst.",
    )
    p.add_argument(
        "--run-filter",
        default=None,
        metavar="REGEX",
        help="When --src is a parent dir, only include child dirs whose name matches this regex.",
    )
    return p.parse_args()


def discover_run_dirs(src: Path) -> List[Path]:
    req = src / "llm_request.json"
    resp = src / "llm_response.json"
    if req.is_file() and resp.is_file():
        return [src.resolve()]
    if not src.is_dir():
        raise FileNotFoundError(f"Not a directory: {src}")
    runs: List[Path] = []
    for child in sorted(src.iterdir()):
        if not child.is_dir():
            continue
        if (child / "llm_request.json").is_file() and (child / "llm_response.json").is_file():
            runs.append(child.resolve())
    if not runs:
        raise FileNotFoundError(
            f"No llm_request.json + llm_response.json under {src} "
            "(expected either a single run dir or a parent of run dirs)."
        )
    return runs


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def response_assistant_content(resp_entry: Dict[str, Any]) -> str:
    body = resp_entry.get("response_body") or {}
    choices = body.get("choices") or []
    if not choices:
        raise KeyError(f"No choices in response for {resp_entry.get('target_key')!r}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise KeyError(f"No message.content for {resp_entry.get('target_key')!r}")
    if not isinstance(content, str):
        content = str(content)
    return content


def request_messages(req_entry: Dict[str, Any]) -> List[Dict[str, str]]:
    body = req_entry.get("request_body") or {}
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise KeyError(f"No request_body.messages for {req_entry.get('target_key')!r}")
    out: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise ValueError(f"Bad message shape in {req_entry.get('target_key')!r}: {m!r}")
        if role == "user":
            content = strip_confidence_from_user_prompt(content)
        out.append({"role": role, "content": content})
    if out[-1]["role"] == "assistant":
        raise ValueError(
            f"Request already contains assistant turn for {req_entry.get('target_key')!r}; "
            "expected system+user only."
        )
    return out


def index_by_target_attempt(entries: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for e in entries:
        tk = e.get("target_key")
        att = e.get("attempt", 0)
        if tk is None:
            LOGGER.warning("Skipping entry without target_key: %s", e)
            continue
        if not isinstance(att, int):
            att = int(att)
        out[(str(tk), att)] = e
    return out


def convert_run(run_dir: Path) -> List[Dict[str, Any]]:
    name = run_dir.name

    req_path = run_dir / "llm_request.json"
    resp_path = run_dir / "llm_response.json"
    requests = load_json_array(req_path)
    responses = load_json_array(resp_path)
    by_resp = index_by_target_attempt(responses)

    # (target_key, attempt) -> one training row; id = "{run_folder}:{target_key}" (no attempt suffix).
    # If multiple attempts exist for the same key, keep the smallest attempt only.
    picked: Dict[str, Tuple[int, List[Dict[str, str]], str]] = {}

    for req in requests:
        tk = str(req.get("target_key", ""))
        att = int(req.get("attempt", 0))
        key = (tk, att)
        resp = by_resp.get(key)
        if resp is None:
            LOGGER.warning(
                "No matching llm_response for target_key=%r attempt=%s in %s",
                tk,
                att,
                run_dir,
            )
            continue
        try:
            msgs = request_messages(req)
            assistant = strip_confidence_from_assistant(response_assistant_content(resp))
        except (KeyError, ValueError) as exc:
            LOGGER.warning("Skip %s (%r, attempt=%s): %s", run_dir.name, tk, att, exc)
            continue

        prev = picked.get(tk)
        if prev is not None and prev[0] <= att:
            continue
        picked[tk] = (att, msgs, assistant)

    rows: List[Dict[str, Any]] = []
    for tk in sorted(picked.keys()):
        _att, msgs, assistant = picked[tk]
        rows.append(
            {
                "id": f"{name}:{tk}",
                "source_run": name,
                "target_key": tk,
                "messages": msgs + [{"role": "assistant", "content": assistant}],
            }
        )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    run_filter = re.compile(args.run_filter) if args.run_filter else None

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    out_path = dst / args.output_name

    run_dirs = discover_run_dirs(src)
    if run_filter is not None:
        run_dirs = [r for r in run_dirs if run_filter.search(r.name)]
    if not run_dirs:
        LOGGER.error("No run directories after filter.")
        sys.exit(1)

    all_rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        part = convert_run(rd)
        LOGGER.info("Run %s -> %d examples", rd.name, len(part))
        all_rows.extend(part)

    # train.jsonl for Trainer: only id + messages (extra keys optional; train.py only uses messages)
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            slim = {"id": row["id"], "messages": row["messages"]}
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    LOGGER.info("Wrote %d lines to %s", len(all_rows), out_path)


if __name__ == "__main__":
    main()
