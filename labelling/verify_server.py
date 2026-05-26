#!/usr/bin/env python3
"""
Local labelling UI for train.jsonl: PDF + Evidence + assistant edit, save under labelling/verified/{YYYY-MM-DD}.json.

**읽기:** `verified_dir` 안의 모든 `YYYY-MM-DD.json`을 읽어 `id`별 최신 검수본을 병합합니다(실행 날짜로 한 파일만 보지 않음).

**쓰기:** Submit 시 기본은 **서버의 달력 오늘** 날짜 파일에 upsert하고, UI/API에서 `save_date`를 지정하면 해당 날짜 파일에 저장합니다.

Submit은 train.jsonl의 행 `id` 기준으로 **upsert**합니다(같은 파일 안에서 같은 id를 다시 저장하면 교체). 여러 날짜 파일이 있으면 파일명(YYYY-MM-DD)이 늦은 쪽·같은 파일 안에서는 뒤에 나온 항목이 우선합니다.

Run from the train project directory (the parent of labelling/):

    uv run python labelling/verify_server.py

Default bind: 0.0.0.0:8765 (reachable from other machines). Locally: http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# verify_server.py lives at train/labelling/ → train dir is parents[1]
_TRAIN_DIR = Path(__file__).resolve().parents[1]
# PDFs live next to train/ under the inference-pipeline repo root (sibling of train/)
_PIPELINE_ROOT = _TRAIN_DIR.parent

DEFAULT_JSONL = _TRAIN_DIR / "dataset" / "train_gt.jsonl"
DEFAULT_PDF_DIR = _PIPELINE_ROOT / "/data1/share/maruchanpark/projects/edinburgh/koreanre/inference-pipeline/학습269건"
DEFAULT_VERIFIED_DIR = _TRAIN_DIR / "labelling" / "verified"

ROWS: List[Dict[str, Any]] = []
CONFIG: Dict[str, Path] = {}
# Latest verification per row id (from all YYYY-MM-DD.json under verified_dir).
VERIFIED_BY_ID: Dict[str, Dict[str, Any]] = {}

_DATE_VERIFIED_JSON = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
_DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class VerifyRequest(BaseModel):
    id: str = Field(..., description="Row id from train.jsonl")
    messages: List[Dict[str, str]] = Field(
        ...,
        description="Full messages including corrected assistant turn",
    )
    save_date: Optional[str] = Field(
        default=None,
        description="YYYY-MM-DD stem for {save_date}.json to upsert into; omit = server calendar today.",
    )

    @field_validator("save_date")
    @classmethod
    def _validate_save_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not str(v).strip():
            return None
        s = str(v).strip()
        if not _DATE_ISO.match(s):
            raise ValueError("save_date must be YYYY-MM-DD")
        return s


def extract_user_content_preamble(user_content: str) -> str:
    """Return user message text before ### Target Field (LLM task instructions)."""
    if not user_content or not str(user_content).strip():
        return ""
    text = str(user_content)
    if "### Target Field" in text:
        head = text.split("### Target Field", 1)[0]
        return head.rstrip()
    return text.strip()


def extract_target_field_block(user_content: str) -> str:
    """Return text under ### Target Field until ### Evidence."""
    m = re.search(
        r"### Target Field\s*\n(.*?)(?=\n### Evidence\b)",
        user_content,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    if "### Target Field" in user_content:
        tail = user_content.split("### Target Field", 1)[1].lstrip("\n\r")
        if "### Evidence" in tail:
            tail = tail.split("### Evidence", 1)[0]
        return tail.strip()
    return ""


def extract_evidence_block(user_content: str) -> str:
    """Return text under ### Evidence until ### Output JSON schema."""
    m = re.search(
        r"### Evidence\s*\n(.*?)(?=\n### Output JSON schema)",
        user_content,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    if "### Evidence" in user_content:
        tail = user_content.split("### Evidence", 1)[1].lstrip("\n\r")
        if "### Output JSON schema" in tail:
            tail = tail.split("### Output JSON schema", 1)[0]
        return tail.strip()
    return ""


def source_run_from_id(row_id: str) -> str:
    if ":" not in row_id:
        return row_id
    return row_id.rsplit(":", 1)[0]


def load_jsonl(path: Path) -> None:
    global ROWS
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    ROWS = rows


def _entries_from_verified_raw(raw: Any) -> List[Dict[str, Any]]:
    """Normalize verified file payload to a list of dict records."""
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        inner = raw.get("entries")
        if isinstance(inner, list):
            return [x for x in inner if isinstance(x, dict)]
        return [raw]
    return []


def _list_verified_json_paths(verified_dir: Path) -> List[Path]:
    if not verified_dir.is_dir():
        return []
    paths = [
        p
        for p in verified_dir.iterdir()
        if p.is_file() and _DATE_VERIFIED_JSON.match(p.name)
    ]
    return sorted(paths)


def refresh_verified_cache() -> int:
    """Rebuild VERIFIED_BY_ID from all YYYY-MM-DD.json files (later file / later row wins)."""
    global VERIFIED_BY_ID
    verified_dir = CONFIG.get("verified_dir")
    if not isinstance(verified_dir, Path):
        VERIFIED_BY_ID = {}
        return 0
    by_id: Dict[str, Dict[str, Any]] = {}
    for path in _list_verified_json_paths(verified_dir):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for rec in _entries_from_verified_raw(raw):
            rid = rec.get("id")
            if rid is not None and str(rid):
                by_id[str(rid)] = rec
    VERIFIED_BY_ID = by_id
    return len(VERIFIED_BY_ID)


def verified_files_payload() -> Dict[str, Any]:
    """List dated verified JSON files (newest first) + server today for UI."""
    verified_dir = CONFIG.get("verified_dir")
    today = datetime.now().strftime("%Y-%m-%d")
    if not isinstance(verified_dir, Path):
        return {"today": today, "verified_dir": "", "files": []}
    items: List[Dict[str, Any]] = []
    for p in sorted(_list_verified_json_paths(verified_dir), reverse=True):
        date = p.stem
        try:
            size_b = p.stat().st_size
        except OSError:
            size_b = 0
        nrec = 0
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            nrec = len(_entries_from_verified_raw(raw))
        except (OSError, json.JSONDecodeError):
            pass
        items.append(
            {
                "date": date,
                "filename": p.name,
                "path": str(p),
                "record_count": nrec,
                "size_bytes": size_b,
            }
        )
    return {
        "today": today,
        "verified_dir": str(verified_dir),
        "files": items,
    }


def build_app() -> FastAPI:
    app = FastAPI(title="RAG-KV train.jsonl verifier", version="0.1.0")

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def root() -> FileResponse:
        index = static_dir / "index.html"
        if not index.is_file():
            raise HTTPException(500, "static/index.html missing")
        return FileResponse(index, media_type="text/html; charset=utf-8")

    @app.get("/api/config")
    def api_config() -> Dict[str, Any]:
        return {
            "jsonl": str(CONFIG["jsonl"]),
            "pdf_dir": str(CONFIG["pdf_dir"]),
            "verified_dir": str(CONFIG["verified_dir"]),
            "total": len(ROWS),
            "verified_count": len(VERIFIED_BY_ID),
            "today": datetime.now().strftime("%Y-%m-%d"),
        }

    @app.get("/api/verified-files")
    def api_verified_files() -> Dict[str, Any]:
        return verified_files_payload()

    @app.post("/api/reload-verified")
    def api_reload_verified() -> Dict[str, Any]:
        n = refresh_verified_cache()
        out = verified_files_payload()
        out["ok"] = True
        out["verified_count"] = n
        return out

    @app.post("/api/reload")
    def api_reload() -> Dict[str, Any]:
        load_jsonl(CONFIG["jsonl"])
        n_verified = refresh_verified_cache()
        return {"ok": True, "total": len(ROWS), "verified_count": n_verified}

    @app.get("/api/row/{index}")
    def api_row(index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(ROWS):
            raise HTTPException(404, "index out of range")
        row = ROWS[index]
        row_id = str(row.get("id", ""))
        base_messages = row.get("messages") or []
        if not isinstance(base_messages, list):
            base_messages = []

        vrec = VERIFIED_BY_ID.get(row_id)
        messages: List[Any] = base_messages
        if isinstance(vrec, dict):
            vm = vrec.get("messages")
            if isinstance(vm, list) and vm:
                messages = vm

        user_content = ""
        assistant = ""
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                user_content = m["content"]
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                assistant = m["content"]
        source_run = source_run_from_id(row_id)
        pdf_path = CONFIG["pdf_dir"] / f"{source_run}.pdf"
        pdf_ok = pdf_path.is_file()
        enc = quote(source_run, safe="")
        pdf_url = f"/api/pdf?source_run={enc}" if pdf_ok else None
        verified_at = None
        if isinstance(vrec, dict):
            va = vrec.get("verified_at")
            if isinstance(va, str):
                verified_at = va
        return {
            "index": index,
            "total": len(ROWS),
            "id": row_id,
            "source_run": source_run,
            "user_content_preamble": extract_user_content_preamble(user_content),
            "target_field": extract_target_field_block(user_content),
            "evidence": extract_evidence_block(user_content),
            "assistant": assistant,
            "messages": messages,
            "pdf_url": pdf_url,
            "pdf_missing": not pdf_ok,
            "verified": vrec is not None,
            "verified_at": verified_at,
        }

    @app.get("/api/pdf")
    def api_pdf(source_run: str = Query(..., description="Decoded run folder / PDF basename without .pdf")):
        name = unquote(source_run)
        pdf_path = (CONFIG["pdf_dir"] / f"{name}.pdf").resolve()
        base = CONFIG["pdf_dir"].resolve()
        if not str(pdf_path).startswith(str(base)) or not pdf_path.is_file():
            raise HTTPException(404, "PDF not found")
        # Starlette defaults content_disposition_type="attachment" → iframe에서 다운로드만 됨.
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=pdf_path.name,
            content_disposition_type="inline",
        )

    @app.post("/api/verify")
    def api_verify(req: VerifyRequest) -> Dict[str, Any]:
        verified_dir = CONFIG["verified_dir"]
        verified_dir.mkdir(parents=True, exist_ok=True)
        date_str = req.save_date or datetime.now().strftime("%Y-%m-%d")
        out_path = verified_dir / f"{date_str}.json"

        record = {
            "verified_at": datetime.now().isoformat(timespec="seconds"),
            "id": req.id,
            "messages": req.messages,
        }

        existing: List[Dict[str, Any]] = []
        if out_path.is_file():
            try:
                raw = json.loads(out_path.read_text(encoding="utf-8"))
                existing = _entries_from_verified_raw(raw)
            except json.JSONDecodeError:
                existing = []

        new_id = str(req.id)
        # Same-day file: collapse to one record per id (last in file wins), then apply this submit.
        by_id_today: Dict[str, Dict[str, Any]] = {}
        for e in existing:
            if not isinstance(e, dict):
                continue
            rid = e.get("id")
            if rid is None or not str(rid):
                continue
            by_id_today[str(rid)] = e
        by_id_today[new_id] = record
        filtered = list(by_id_today.values())
        out_path.write_text(
            json.dumps(filtered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        refresh_verified_cache()
        after = VERIFIED_BY_ID.get(new_id)
        eff_at = None
        if isinstance(after, dict):
            va = after.get("verified_at")
            if isinstance(va, str):
                eff_at = va
        return {
            "ok": True,
            "path": str(out_path),
            "save_date": date_str,
            "record_count": len(filtered),
            "count_today": len(filtered),
            "upserted": True,
            "verified_at": record["verified_at"],
            "effective_verified": after is not None,
            "effective_verified_at": eff_at,
        }

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="Path to train.jsonl",
    )
    p.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help="Directory containing {source_run}.pdf",
    )
    p.add_argument(
        "--verified-dir",
        type=Path,
        default=DEFAULT_VERIFIED_DIR,
        help="Directory for YYYY-MM-DD.json outputs (id upsert per day; status merges all dates)",
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Listen address (0.0.0.0 = all interfaces; use 127.0.0.1 for local only).",
    )
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global CONFIG
    CONFIG = {
        "jsonl": args.jsonl.expanduser().resolve(),
        "pdf_dir": args.pdf_dir.expanduser().resolve(),
        "verified_dir": args.verified_dir.expanduser().resolve(),
    }
    load_jsonl(CONFIG["jsonl"])
    refresh_verified_cache()
    import uvicorn

    app = build_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
