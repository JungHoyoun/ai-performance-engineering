#!/usr/bin/env python3
"""Extract summary metrics from Nsight Systems reports (CSV or .nsys-rep)."""

import argparse
import csv
import io
import pathlib
import subprocess
import sys
import tempfile
from typing import Iterable, List, Dict


def _read_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def _run_nsys_stats(rep_path: pathlib.Path) -> List[Dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="nsys_stats_") as tmp_dir:
        output_prefix = pathlib.Path(tmp_dir) / "report"
        command = [
            "nsys",
            "stats",
            "--report",
            "nvtx_sum",
            "--format",
            "csv",
            "--force-export",
            "true",
            "--output",
            str(output_prefix),
            str(rep_path),
        ]
        try:
            subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)  # 60s - extraction can be slow for large files
        except subprocess.TimeoutExpired:
            raise SystemExit(f"nsys stats timed out after 60s for {rep_path} (file may be very large)")
        except FileNotFoundError as exc:
            raise SystemExit("nsys binary not found on PATH; install Nsight Systems to extract summaries") from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"nsys stats failed for {rep_path}: {exc.stderr.strip()}") from exc

        csv_files = sorted(pathlib.Path(tmp_dir).glob("report*nvtx_sum.csv"))
        if not csv_files:
            return []
        csv_path = csv_files[0]
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            return [row for row in reader]


def harvest(path: pathlib.Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".csv":
        rows = _read_csv(path)
    else:
        rows = _run_nsys_stats(path)

    extracted: List[Dict[str, str]] = []
    for row in rows:
        if not row:
            continue
        section = row.get("Section") or row.get("Style") or ""

        metric_name = (
            row.get("Metric Name")
            or row.get("Name")
            or row.get("Range")
        )
        value = (
            row.get("Metric Value")
            or row.get("Value")
            or row.get("Total Time (ns)")
        )

        if metric_name and value:
            extracted.append({"section": section, "metric": metric_name, "value": value})

        # Capture percentage columns as separate metrics when available
        pct_value = row.get("Time (%)")
        if metric_name and pct_value:
            extracted.append({"section": section, "metric": f"{metric_name}_pct", "value": pct_value})
    return extracted


def process(patterns: Iterable[str]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    seen_paths = set()
    
    for pattern in patterns:
        path = pathlib.Path(pattern)
        # Check if pattern is a direct file path (no glob characters and exists)
        # Or if it's an absolute path (likely a direct file path)
        if path.is_absolute() or ("*" not in pattern and "?" not in pattern and "[" not in pattern):
            # Try as direct file path
            if path.exists() and path.is_file():
                resolved = path.resolve()
                if resolved not in seen_paths:
                    seen_paths.add(resolved)
                    metrics = harvest(path)
                    tag = path.stem
                    for entry in metrics:
                        record = {"tag": tag}
                        record.update(entry)
                        output.append(record)
            continue
        
        # Try as glob pattern (relative to current directory)
        for candidate in pathlib.Path().glob(pattern):
            if not candidate.exists() or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                metrics = harvest(candidate)
                tag = candidate.stem
                for entry in metrics:
                    record = {"tag": tag}
                    record.update(entry)
                    output.append(record)
    return output


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Extract Nsight Systems summary metrics to CSV")
    parser.add_argument("patterns", nargs="+", help="Glob pattern(s) or file path(s) to .nsys-rep or CSV files")
    parser.add_argument(
        "--output",
        default="artifacts/runs/analysis/nsys_summary.csv",
        help="Destination CSV file (default: artifacts/runs/analysis/nsys_summary.csv)",
    )
    args = parser.parse_args(argv)

    rows = process(args.patterns)
    if not rows:
        print("No Nsight Systems metrics found", file=sys.stderr)
        return 1

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tag", "section", "metric", "value"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
