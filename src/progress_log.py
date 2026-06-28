"""Small append-only progress logger for long optimization runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_progress(path: str | Path | None, message: str, data: dict[str, Any] | None = None) -> None:
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }
    if data:
        payload["data"] = data
    line = json.dumps(payload, ensure_ascii=False, default=str)
    print(f"[progress] {line}", flush=True)
    if path is None:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
