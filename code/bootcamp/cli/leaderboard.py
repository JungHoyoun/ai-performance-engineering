from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import fcntl


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LEADERBOARD_PATH = DATA_DIR / "leaderboard_local.json"
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_structure() -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "players": {}}


@contextmanager
def _locked_file(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            contents = handle.read()
            data = json.loads(contents) if contents.strip() else _default_structure()
            if data.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported leaderboard schema {data.get('schema_version')}; expected {SCHEMA_VERSION}."
                )
            yield data, handle
            handle.seek(0)
            handle.truncate()
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_leaderboard() -> Dict[str, Any]:
    if not LEADERBOARD_PATH.exists():
        return _default_structure()
    with LEADERBOARD_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def record_run(
    player_id: str,
    level: str,
    metrics: Dict[str, Any],
    *,
    hardware: Optional[Dict[str, Any]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a run for the specified player and return the updated entry."""
    with _locked_file(LEADERBOARD_PATH) as (data, _handle):
        players = data.setdefault("players", {})
        player_entry = players.setdefault(player_id, {"runs": []})
        run_entry: Dict[str, Any] = {
            "level": level,
            "metrics": metrics,
            "timestamp": _now(),
        }
        if hardware:
            run_entry["hardware"] = hardware
        if note:
            run_entry["note"] = note
        player_entry["runs"].append(run_entry)
        player_entry["runs"] = sorted(player_entry["runs"], key=lambda r: r["timestamp"], reverse=True)
        players[player_id] = player_entry
        return run_entry


def latest_runs(data: Dict[str, Any], *, limit: int = 5) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for player_id, entry in data.get("players", {}).items():
        runs = entry.get("runs", [])[:limit]
        snapshot[player_id] = runs
    return snapshot


__all__ = [
    "DATA_DIR",
    "LEADERBOARD_PATH",
    "SCHEMA_VERSION",
    "latest_runs",
    "load_leaderboard",
    "record_run",
]
