#!/usr/bin/env python3
"""Restore queued bench/profiling runs from a snapshot JSON."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List


def _load_snapshot(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "restore_commands" not in data:
        raise KeyError("Snapshot missing restore_commands")
    if not isinstance(data["restore_commands"], list):
        raise TypeError("restore_commands must be a list")
    return data


def _spawn_command(cmd: str, cwd: str) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        executable="/bin/bash",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Path to snapshot JSON produced by scripts/queue_snapshot.py.",
    )
    args = parser.parse_args()

    snapshot = _load_snapshot(args.snapshot)
    restore_commands: List[str] = snapshot["restore_commands"]
    if not restore_commands:
        raise RuntimeError("No restore commands in snapshot")

    cwd = snapshot.get("cwd") or os.getcwd()
    procs: List[subprocess.Popen] = []

    for cmd in restore_commands:
        print(f"[queue_restore] launching: {cmd}")
        procs.append(_spawn_command(cmd, cwd))

    print(f"[queue_restore] spawned {len(procs)} process(es) (cwd={cwd})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
