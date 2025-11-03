from __future__ import annotations

import argparse
import importlib
import sys
from typing import Callable, Dict


LEVEL_MODULES: Dict[str, str] = {
    "ch1": "bootcamp.ch1.game_hooks",
    "ch7": "bootcamp.ch7.game_hooks",
    "ch16": "bootcamp.ch16.game_hooks",
}


def _resolve_level(level: str) -> Callable[[], None]:
    if level not in LEVEL_MODULES:
        available = ", ".join(sorted(LEVEL_MODULES))
        raise SystemExit(f"Unsupported level '{level}'. Available levels: {available}")

    module = importlib.import_module(LEVEL_MODULES[level])
    for attr in ("run_for_profiler", "run_profiler", "profile_runner"):
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
    raise SystemExit(f"Level '{level}' does not expose a profiler runner.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bootcamp Nsight Compute runner.")
    parser.add_argument("--level", required=True, help="Level identifier (e.g. ch1, ch7, ch16).")
    args = parser.parse_args(argv)

    runner = _resolve_level(args.level)
    runner()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
