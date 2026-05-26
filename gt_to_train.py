#!/usr/bin/env python3
"""
Merge VIA-style gt.json (bbox + per-class value text) and optional answer_sheet.xlsx
into RAG-KV train.jsonl labels.

For each train row:
  - Resolve the gt image by document stem (from ``id``) + page (from Evidence / chunk ids).
  - Build per-class text from gt ``regions`` (value boxes), reading order by (y, x).
  - Map ``보상한도금`` + ``공제금`` in gt to train field ``보상한도_공제금`` (newline-joined,
    보상한도금 first).
  - Assistant JSON (gt): copy existing ``verbatim_quote`` to ``llm_prediction`` if not already
    present; set ``verbatim_quote`` from gt. Other keys (e.g. ``evidence_chunk_ids``) are kept.
  - Assistant JSON (``--answer-sheet-xlsx``): set ``answer_sheet`` from the spreadsheet cell
    for the same document stem (``file_name`` without ``.pdf``) and target field column.
    Train field ``담보내용`` maps to sheet column ``담보_내용``; ``임의출재율`` / ``임의수수료율``
    map to ``임의_출재율`` / ``임의_수수료율``. Requires ``openpyxl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Train ``id`` suffix (Target Field key) -> ``answer_sheet.xlsx`` column header.
_TRAIN_FIELD_TO_SHEET_COL: Dict[str, str] = {
    "담보내용": "담보_내용",
    "임의출재율": "임의_출재율",
    "임의수수료율": "임의_수수료율",
}

LOGGER = logging.getLogger(__name__)

_RE_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_RE_CHUNK_PAGE = re.compile(r"\[chunk_id=(\d+)\s+page=(\d+)\]")


def repair_invalid_json_string_escapes(json_text: str) -> str:
    """
    Best-effort fix for invalid ``\\`` sequences inside JSON string literals (common in LLM output).

    Valid JSON escapes: ``\"``, ``\\\\``, ``\\/``, ``\\b``, ``\\f``, ``\\n``, ``\\r``, ``\\t``, ``\\uXXXX``.
    Any other ``\\`` before another character becomes ``\\\\`` so :func:`json.loads` can succeed.
    """
    out: List[str] = []
    i = 0
    n = len(json_text)
    in_string = False

    def preceding_out_backslashes() -> int:
        bs = 0
        j = len(out) - 1
        while j >= 0 and out[j] == "\\":
            bs += 1
            j -= 1
        return bs

    while i < n:
        ch = json_text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                if preceding_out_backslashes() % 2 == 0:
                    in_string = True
            i += 1
            continue

        # Inside a JSON string value.
        if ch == '"':
            out.append(ch)
            if preceding_out_backslashes() % 2 == 0:
                in_string = False
            i += 1
            continue

        if ch == "\\" and i + 1 < n:
            nxt = json_text[i + 1]
            if nxt in '"\\/bfnrt':
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            if nxt == "u" and i + 5 < n:
                hexpart = json_text[i + 2 : i + 6]
                if len(hexpart) == 4 and all(c in "0123456789abcdefABCDEF" for c in hexpart):
                    out.append(json_text[i : i + 6])
                    i += 6
                    continue
            out.append("\\\\")
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def loads_assistant_obj(raw: str) -> Dict[str, Any]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = json.loads(repair_invalid_json_string_escapes(raw))
    if not isinstance(obj, dict):
        raise ValueError("Assistant JSON root must be an object.")
    return obj


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gt-json",
        type=Path,
        default=root / "dataset" / "gt.json",
        help="VIA export JSON (object keyed by filename+size).",
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        default=root / "dataset" / "train.jsonl",
        help="Input JSONL (id + messages).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Default: sibling train.from_gt.jsonl next to --train-jsonl.",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --train-jsonl (same as --output pointing to that file).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any row has no matching gt image.",
    )
    p.add_argument(
        "--answer-sheet-xlsx",
        type=Path,
        default=None,
        help="Optional Excel (first sheet, ``file_name`` column). Adds ``answer_sheet`` to each "
        "assistant JSON from the row matching the train document stem.",
    )
    return p.parse_args()


def load_gt(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object (VIA dict keyed by image id).")
    return data


def build_gt_index(gt: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Index by (document_stem, page) -> full image entry (filename, size, regions, ...)."""
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for _key, entry in gt.items():
        if not isinstance(entry, dict):
            continue
        fn = entry.get("filename")
        if not isinstance(fn, str):
            continue
        m = re.search(r"^(.*)_page(\d{6})\.png", fn)
        if not m:
            LOGGER.warning("Skip gt entry with unexpected filename: %r", fn[:120])
            continue
        stem, page_s = m.group(1), m.group(2)
        page = int(page_s)
        dup = index.get((stem, page))
        if dup is not None:
            LOGGER.warning(
                "Duplicate gt for stem=%r page=%s (keeping size=%s, also saw size=%s)",
                stem,
                page,
                entry.get("size"),
                dup.get("size"),
            )
        index[(stem, page)] = entry
    return index


def parse_train_id(row_id: str) -> Tuple[str, str]:
    if ":" not in row_id:
        raise ValueError(f"Bad id (no ':'): {row_id!r}")
    stem, field = row_id.rsplit(":", 1)
    return stem, field


def infer_page_from_evidence(user_text: str, chunk_ids: List[Any]) -> int:
    """Page of the first evidence_chunk_id that appears in the user Evidence block."""
    for cid in chunk_ids:
        try:
            cid_i = int(cid)
        except (TypeError, ValueError):
            continue
        m = re.search(rf"\[chunk_id={cid_i}\s+page=(\d+)\]", user_text)
        if m:
            return int(m.group(1))
    # Fallback: first chunk line in document order
    for m in _RE_CHUNK_PAGE.finditer(user_text):
        return int(m.group(2))
    LOGGER.warning("Could not infer page from Evidence; defaulting to 1.")
    return 1


def regions_to_class_texts(regions: List[Dict[str, Any]]) -> Dict[str, str]:
    """class -> joined text; multiple boxes sorted by (y, x), joined with newlines."""
    buckets: Dict[str, List[Tuple[int, int, str]]] = {}
    for r in regions:
        ra = r.get("region_attributes") or {}
        cls = ra.get("class")
        if not cls:
            continue
        sc = ra.get("sub_class")
        if sc is not None and str(sc).lower() != "value":
            continue
        text = ra.get("text")
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        sa = r.get("shape_attributes") or {}
        y = int(sa.get("y", 0))
        x = int(sa.get("x", 0))
        buckets.setdefault(str(cls), []).append((y, x, text))
    out: Dict[str, str] = {}
    for cls, items in buckets.items():
        items.sort(key=lambda t: (t[0], t[1]))
        out[cls] = "\n".join(t[2] for t in items)
    return out


def verbatim_for_field(class_texts: Dict[str, str], field_key: str) -> str:
    if field_key == "보상한도_공제금":
        a = (class_texts.get("보상한도금") or "").strip()
        b = (class_texts.get("공제금") or "").strip()
        if a and b:
            return f"{a}\n{b}"
        return a or b
    return class_texts.get(field_key, "") or ""


def split_assistant_fence(content: str) -> Tuple[str, bool]:
    raw = content.strip()
    m = _RE_FENCE.match(raw)
    if m:
        return m.group(1).strip(), True
    return raw, False


def format_answer_sheet_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def sheet_column_for_train_field(field_key: str) -> str:
    return _TRAIN_FIELD_TO_SHEET_COL.get(field_key, field_key)


def load_answer_sheet_index(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        import openpyxl  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Reading --answer-sheet-xlsx requires openpyxl. Install with: pip install openpyxl"
        ) from e

    wb = openpyxl.load_workbook(path.expanduser().resolve(), read_only=True, data_only=True)
    try:
        ws = wb.active
        rows = ws.iter_rows(values_only=True)
        try:
            header_row = next(rows)
        except StopIteration:
            return {}
        headers: List[str] = []
        for h in header_row:
            headers.append("" if h is None else str(h))
        if not headers or headers[0] != "file_name":
            LOGGER.warning(
                "answer_sheet: expected first column 'file_name', got %r. Proceeding anyway.",
                headers[0] if headers else None,
            )
        index: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if not row or row[0] is None:
                continue
            stem = str(row[0]).strip()
            if stem.endswith(".pdf"):
                stem = stem[: -len(".pdf")]
            row_dict = {headers[i]: row[i] if i < len(row) else None for i in range(len(headers))}
            dup = stem in index
            if dup:
                LOGGER.warning("Duplicate answer_sheet file_name stem=%r (overwriting).", stem[:120])
            index[stem] = row_dict
        return index
    finally:
        wb.close()


def merge_assistant_json(
    content: str,
    *,
    gt_verbatim: Optional[str] = None,
    answer_sheet: Optional[str] = None,
) -> str:
    if gt_verbatim is None and answer_sheet is None:
        return content

    raw, fenced = split_assistant_fence(content)
    obj = loads_assistant_obj(raw)

    use_pretty = "\n" in content.strip() or fenced

    if gt_verbatim is not None:
        old_vq = obj.get("verbatim_quote", "")
        if not isinstance(old_vq, str):
            old_vq = str(old_vq)

        if "llm_prediction" not in obj:
            obj["llm_prediction"] = old_vq
        obj["verbatim_quote"] = gt_verbatim

    if answer_sheet is not None:
        obj["answer_sheet"] = answer_sheet

    if use_pretty:
        body = json.dumps(obj, ensure_ascii=False, indent=2)
    else:
        body = json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
    if fenced:
        return "```json\n" + body + "\n```"
    return body


def pick_user_assistant(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    user = None
    assistant = None
    for m in messages:
        if m.get("role") == "user":
            user = m.get("content")
        elif m.get("role") == "assistant":
            assistant = m.get("content")
    if not isinstance(user, str) or not isinstance(assistant, str):
        raise ValueError("messages must include string user and assistant contents.")
    return user, assistant


def process_row(
    row: Dict[str, Any],
    gt_index: Dict[Tuple[str, int], Dict[str, Any]],
    strict: bool,
    answer_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Returns (updated_row, True) if a matching gt image was found (verbatim updated from gt)."""
    row_id = row.get("id")
    if not isinstance(row_id, str):
        raise ValueError("Row missing string id.")
    stem, field = parse_train_id(row_id)
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Row {row_id!r}: messages must be a list.")

    user_text, assistant_text = pick_user_assistant(messages)
    raw_asst, _f = split_assistant_fence(assistant_text)
    asst_obj = loads_assistant_obj(raw_asst)
    chunk_ids = asst_obj.get("evidence_chunk_ids") or []
    if not isinstance(chunk_ids, list):
        chunk_ids = []

    page = infer_page_from_evidence(user_text, chunk_ids)
    gt_entry = gt_index.get((stem, page))
    gt_verbatim: Optional[str] = None
    matched_gt = False
    if gt_entry is None:
        msg = f"No gt image for stem={stem!r} page={page} (id={row_id!r})."
        if strict:
            raise FileNotFoundError(msg)
        LOGGER.warning("%s Skipping gt verbatim update.", msg)
    else:
        regions = gt_entry.get("regions") or []
        if not isinstance(regions, list):
            regions = []
        class_texts = regions_to_class_texts(regions)
        gt_verbatim = verbatim_for_field(class_texts, field)
        matched_gt = True

    answer_sheet_val: Optional[str] = None
    if answer_index is not None:
        sheet_row = answer_index.get(stem)
        if sheet_row is None:
            LOGGER.warning("No answer_sheet row for stem=%r (id=%r).", stem[:120], row_id)
            answer_sheet_val = ""
        else:
            col = sheet_column_for_train_field(field)
            if col not in sheet_row:
                LOGGER.warning(
                    "answer_sheet: missing column %r for field=%r (id=%r).",
                    col,
                    field,
                    row_id,
                )
                answer_sheet_val = ""
            else:
                answer_sheet_val = format_answer_sheet_cell(sheet_row.get(col))

    if gt_verbatim is None and answer_sheet_val is None:
        return row, False

    new_asst = merge_assistant_json(
        assistant_text,
        gt_verbatim=gt_verbatim,
        answer_sheet=answer_sheet_val,
    )
    new_messages: List[Dict[str, Any]] = []
    replaced = False
    for m in messages:
        if m.get("role") == "assistant":
            nm = dict(m)
            nm["content"] = new_asst
            new_messages.append(nm)
            replaced = True
        else:
            new_messages.append(dict(m))
    if not replaced:
        raise ValueError(f"Row {row_id!r}: no assistant message to update.")

    out = dict(row)
    out["messages"] = new_messages
    return out, matched_gt


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    gt_path = args.gt_json.expanduser().resolve()
    train_path = args.train_jsonl.expanduser().resolve()
    if args.in_place:
        out_path = train_path
    elif args.output is not None:
        out_path = args.output.expanduser().resolve()
    else:
        out_path = train_path.with_name("train.from_gt.jsonl")

    gt = load_gt(gt_path)
    gt_index = build_gt_index(gt)
    LOGGER.info("Loaded gt: %d images, %d (stem,page) index entries.", len(gt), len(gt_index))

    answer_index: Optional[Dict[str, Dict[str, Any]]] = None
    if args.answer_sheet_xlsx is not None:
        ans_path = args.answer_sheet_xlsx.expanduser().resolve()
        answer_index = load_answer_sheet_index(ans_path)
        LOGGER.info("Loaded answer_sheet: %d rows from %s.", len(answer_index), ans_path)

    n_matched = 0
    n_unmatched = 0
    out_lines: List[str] = []
    with train_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                LOGGER.error("Line %s: invalid JSON: %s", lineno, e)
                raise
            try:
                new_row, matched = process_row(
                    row,
                    gt_index,
                    strict=args.strict,
                    answer_index=answer_index,
                )
            except FileNotFoundError:
                raise
            except Exception as e:
                LOGGER.error("Line %s (id=%r): %s", lineno, row.get("id"), e)
                raise
            if matched:
                n_matched += 1
            else:
                n_unmatched += 1
            out_lines.append(json.dumps(new_row, ensure_ascii=False) + "\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)

    LOGGER.info(
        "Wrote %d lines to %s (gt stem+page matched: %d, no gt stem+page: %d).",
        len(out_lines),
        out_path,
        n_matched,
        n_unmatched,
    )


if __name__ == "__main__":
    main()
