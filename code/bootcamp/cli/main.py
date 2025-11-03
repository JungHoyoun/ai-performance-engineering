from __future__ import annotations

import argparse
import getpass
import json
import os
from typing import Any, Dict, List

import torch

from .leaderboard import latest_runs, load_leaderboard, record_run


def _default_player() -> str:
    return os.getenv("BOOTCAMP_PLAYER") or getpass.getuser()


def _detect_hardware() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"gpu": "cpu", "cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "cuda_available": True,
        "total_memory_bytes": props.total_memory,
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for caster in (int, float):
        try:
            coerced = caster(value)
            return coerced
        except ValueError:
            continue
    return value


def _parse_kv(pairs: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"Invalid key=value pair: '{pair}'")
        key, value = pair.split("=", 1)
        parsed[key] = _coerce_value(value)
    return parsed


def cmd_status(args: argparse.Namespace) -> None:
    data = load_leaderboard()
    snapshot = data if args.full else latest_runs(data, limit=args.limit)
    if args.json:
        print(json.dumps(snapshot, indent=2))
        return
    players = data.get("players", {}) if args.full else snapshot
    if not players:
        print("No leaderboard data recorded yet.")
        return
    for player_id, entry in players.items():
        print(f"Player: {player_id}")
        runs = entry.get("runs", []) if isinstance(entry, dict) else entry
        for run in runs[: args.limit]:
            metrics = run.get("metrics", {})
            level = run.get("level", "unknown")
            ts = run.get("timestamp", "n/a")
            print(f"  - [{ts}] Level {level}: {metrics}")


def cmd_record(args: argparse.Namespace) -> None:
    metrics = _parse_kv(args.metric)
    hardware = None if args.skip_hardware else _detect_hardware()
    run_entry = record_run(
        player_id=args.player,
        level=args.level,
        metrics=metrics,
        hardware=hardware,
        note=args.note,
    )
    print(json.dumps(run_entry, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bootcamp", description="Inference Empire CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("leaderboard", help="Inspect leaderboard data.")
    sub_status = status_parser.add_subparsers(dest="leaderboard_cmd", required=True)
    status_view = sub_status.add_parser("status", help="Show recent leaderboard entries.")
    status_view.add_argument("--limit", type=int, default=5, help="Number of runs per player to show.")
    status_view.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    status_view.add_argument("--full", action="store_true", help="Show entire leaderboard file.")
    status_view.set_defaults(func=cmd_status)

    record_parser = subparsers.add_parser("record", help="Record a new run entry.")
    record_parser.add_argument("--level", required=True, help="Level identifier (e.g. ch1).")
    record_parser.add_argument(
        "--metric",
        action="append",
        default=[],
        required=True,
        help="Metric key=value pair (repeatable).",
    )
    record_parser.add_argument("--player", default=_default_player(), help="Player identifier.")
    record_parser.add_argument("--note", help="Optional note for the run.")
    record_parser.add_argument("--skip-hardware", action="store_true", help="Do not record hardware info.")
    record_parser.set_defaults(func=cmd_record)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
