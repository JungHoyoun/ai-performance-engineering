#!/usr/bin/env python3
"""
Script to run full benchmarks on EVERY chapter and lab,
ONE at a time, sequentially with TRUE ISOLATION.

Requirements:
- NO LLM analysis
- Sequential execution (one chapter/lab at a time)
- NO parallel execution
- Kill all GPU processes between benchmarks for true isolation
"""

import subprocess
import sys
import json
import time
import argparse
import os
import signal
import fnmatch
from pathlib import Path
from datetime import datetime


def kill_gpu_processes(exclude_pids=None, log_file=None):
    """Kill all processes using the GPU except excluded PIDs.
    
    This ensures true isolation between benchmarks by clearing any
    lingering GPU processes from profilers or previous runs.
    """
    exclude_pids = exclude_pids or set()
    # Add current process and parent to exclusions
    exclude_pids.add(os.getpid())
    exclude_pids.add(os.getppid())
    
    try:
        # Get PIDs of processes using GPU
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return 0
        
        pids_to_kill = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    pid = int(line.strip())
                    if pid not in exclude_pids:
                        pids_to_kill.append(pid)
                except ValueError:
                    continue
        
        killed = 0
        for pid in pids_to_kill:
            try:
                # Try SIGTERM first
                os.kill(pid, signal.SIGTERM)
                killed += 1
                msg = f"  Killed GPU process {pid} (SIGTERM)"
                print(msg)
                if log_file:
                    log_file.write(msg + "\n")
            except ProcessLookupError:
                pass  # Process already dead
            except PermissionError:
                msg = f"  WARNING: No permission to kill GPU process {pid}"
                print(msg)
                if log_file:
                    log_file.write(msg + "\n")
        
        # Wait a bit for processes to die
        if killed > 0:
            time.sleep(2)
            
            # Force kill any remaining
            for pid in pids_to_kill:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        
        return killed
    except Exception as e:
        msg = f"  WARNING: Failed to check/kill GPU processes: {e}"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
        return 0

# Get the list of all chapters and labs
def get_all_chapters_and_labs():
    """Get list of all chapters and labs from the system."""
    try:
        result = subprocess.run(
            ["python", "-m", "cli.aisp", "bench", "list-chapters"],
            capture_output=True,
            text=True,
            check=True
        )
        chapters = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return chapters
    except subprocess.CalledProcessError as e:
        print(f"Error getting chapters list: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


def filter_chapters(chapters, chapter_patterns=None, lab_patterns=None, chapter_list=None, lab_list=None):
    """
    Filter chapters and labs based on wildcard patterns or explicit lists.
    
    Args:
        chapters: List of all available chapters/labs
        chapter_patterns: List of wildcard patterns for chapters (e.g., ['ch1*', 'ch2*'])
        lab_patterns: List of wildcard patterns for labs (e.g., ['labs/*'])
        chapter_list: Explicit list of chapter names to include
        lab_list: Explicit list of lab names to include
    
    Returns:
        Filtered list of chapters/labs
    """
    filtered = []
    
    # If no filters specified, return all chapters
    if not chapter_patterns and not lab_patterns and not chapter_list and not lab_list:
        return chapters
    
    # Process explicit lists first (they take precedence)
    if chapter_list:
        for ch in chapter_list:
            if ch in chapters and ch not in filtered:
                filtered.append(ch)
    
    if lab_list:
        for lab in lab_list:
            if lab in chapters and lab not in filtered:
                filtered.append(lab)
    
    # Process wildcard patterns
    if chapter_patterns:
        for pattern in chapter_patterns:
            for ch in chapters:
                if fnmatch.fnmatch(ch, pattern) and ch not in filtered:
                    filtered.append(ch)
    
    if lab_patterns:
        for pattern in lab_patterns:
            for lab in chapters:
                if fnmatch.fnmatch(lab, pattern) and lab not in filtered:
                    filtered.append(lab)
    
    return filtered


def run_benchmark_for_chapter(chapter, log_file, update_expectations=False, accept_regressions=False, 
                              cold_start=False, reproducible=False, suite_timeout=None, timeout_multiplier=None,
                              profile="deep_dive", kill_gpu=True, skip_verify=False, verify_phase=None):
    """
    Run benchmark for a single chapter/lab with TRUE ISOLATION.
    
    Args:
        chapter: Chapter/lab name (e.g., 'ch01', 'labs/decode_optimization')
        log_file: File handle to write logs to
        update_expectations: If True, add --update-expectations flag
        accept_regressions: If True, add --accept-regressions flag
        cold_start: If True, add --cold-start flag (reset GPU state between benchmarks)
        reproducible: If True, add --reproducible flag (set seeds to 42)
        suite_timeout: Optional timeout in seconds for the suite
        timeout_multiplier: Optional multiplier for benchmark timeouts
        profile: Profiling level ('none', 'minimal', 'deep_dive', 'roofline')
        kill_gpu: If True, kill all GPU processes before running benchmark
        skip_verify: If True, skip output verification
        verify_phase: Verification enforcement phase ('detect', 'quarantine', 'gate')
    """
    print(f"\n{'='*80}")
    print(f"Starting benchmark for: {chapter}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Starting benchmark for: {chapter}\n")
    log_file.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'='*80}\n")
    log_file.flush()
    
    # Kill GPU processes for true isolation
    if kill_gpu:
        print("  Cleaning up GPU processes for isolation...")
        log_file.write("  Cleaning up GPU processes for isolation...\n")
        log_file.flush()
        killed = kill_gpu_processes(log_file=log_file)
        if killed > 0:
            print(f"  Killed {killed} lingering GPU process(es)")
            log_file.write(f"  Killed {killed} lingering GPU process(es)\n")
            log_file.flush()
            time.sleep(2)  # Give GPU time to fully release resources
    
    start_time = time.time()
    
    try:
        # Run benchmark with specified profiling level, NO LLM analysis (default is False)
        cmd = [
            "python", "-m", "cli.aisp", "bench", "run",
            "--targets", chapter,
            "--profile", profile
            # Note: llm_analysis defaults to False, so we don't need to disable it
        ]
        
        # Add expectation update flags if requested
        if update_expectations:
            cmd.append("--update-expectations")
        if accept_regressions:
            cmd.append("--accept-regressions")
        if cold_start:
            cmd.append("--cold-start")
        if reproducible:
            cmd.append("--reproducible")
        if suite_timeout is not None:
            cmd.extend(["--suite-timeout", str(suite_timeout)])
        if timeout_multiplier is not None:
            cmd.extend(["--timeout-multiplier", str(timeout_multiplier)])
        if skip_verify:
            cmd.append("--skip-verify")
        if verify_phase is not None:
            cmd.extend(["--verify-phase", verify_phase])
        
        print(f"Command: {' '.join(cmd)}")
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.flush()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't fail on non-zero exit, we'll handle it
        )
        
        duration = time.time() - start_time
        
        print(f"\nReturn code: {result.returncode}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        log_file.write(f"\nReturn code: {result.returncode}\n")
        log_file.write(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)\n")
        log_file.write(f"\n--- STDOUT ---\n{result.stdout}\n")
        log_file.write(f"\n--- STDERR ---\n{result.stderr}\n")
        log_file.flush()
        
        if result.returncode != 0:
            print(f"WARNING: Benchmark for {chapter} returned non-zero exit code!")
            log_file.write(f"WARNING: Benchmark for {chapter} returned non-zero exit code!\n")
            log_file.flush()
            return False
        else:
            print(f"SUCCESS: Benchmark for {chapter} completed successfully!")
            log_file.write(f"SUCCESS: Benchmark for {chapter} completed successfully!\n")
            log_file.flush()
            return True
            
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"ERROR running benchmark for {chapter}: {str(e)}"
        print(error_msg)
        log_file.write(f"{error_msg}\n")
        log_file.write(f"Duration: {duration:.2f} seconds\n")
        log_file.flush()
        return False

def main():
    """Main function to run benchmarks sequentially."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run full benchmarks with deep profiling on every chapter and lab sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--update-expectations",
        action="store_true",
        help="Force-write observed metrics into expectation files (overrides regressions)"
    )
    parser.add_argument(
        "--accept-regressions",
        action="store_true",
        help="Update expectation files when improvements are detected instead of flagging regressions"
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Reset GPU state between benchmarks for cold start measurements"
    )
    parser.add_argument(
        "--reproducible",
        action="store_true",
        help="Enable reproducible mode: set all seeds to 42 and force deterministic algorithms"
    )
    parser.add_argument(
        "--suite-timeout",
        type=int,
        default=None,
        help="Suite timeout in seconds (default: 14400 = 4 hours, 0 = disabled)"
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=None,
        help="Multiply all benchmark timeouts by this factor (e.g., 2.0 = double all timeouts)"
    )
    parser.add_argument(
        "--profile",
        choices=["none", "minimal", "deep_dive", "roofline"],
        default="deep_dive",
        help="Profiling level (default: deep_dive). Use 'none' for fastest runs without profiling."
    )
    parser.add_argument(
        "--no-kill-gpu",
        action="store_true",
        help="Don't kill GPU processes between benchmarks (faster but less isolated)"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip output verification (useful for initial runs before golden outputs are cached)"
    )
    parser.add_argument(
        "--verify-phase",
        choices=["detect", "quarantine", "gate"],
        default=None,
        help="Verification enforcement phase: 'detect' (report only), 'quarantine' (exclude non-compliant), 'gate' (strict, default)"
    )
    parser.add_argument(
        "--chapter-patterns",
        type=str,
        nargs="+",
        default=None,
        help="Wildcard patterns for chapters to include (e.g., 'ch1*' 'ch2*')"
    )
    parser.add_argument(
        "--lab-patterns",
        type=str,
        nargs="+",
        default=None,
        help="Wildcard patterns for labs to include (e.g., 'labs/*' 'labs/decode*')"
    )
    parser.add_argument(
        "--chapters",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of chapter names to include (e.g., 'ch01' 'ch02')"
    )
    parser.add_argument(
        "--labs",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of lab names to include (e.g., 'labs/decode_optimization')"
    )
    args = parser.parse_args()
    
    kill_gpu = not args.no_kill_gpu
    
    print("="*80)
    print("Sequential Benchmark Runner with TRUE ISOLATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Profile: {args.profile}")
    print(f"Kill GPU processes between runs: {kill_gpu}")
    if args.update_expectations:
        print("Mode: --update-expectations enabled")
    if args.accept_regressions:
        print("Mode: --accept-regressions enabled")
    if args.cold_start:
        print("Mode: --cold-start enabled")
    if args.reproducible:
        print("Mode: --reproducible enabled")
    if args.suite_timeout is not None:
        print(f"Suite timeout: {args.suite_timeout} seconds")
    if args.timeout_multiplier is not None:
        print(f"Timeout multiplier: {args.timeout_multiplier}x")
    if args.skip_verify:
        print("Mode: --skip-verify enabled (skipping output verification)")
    if args.verify_phase:
        print(f"Verify phase: {args.verify_phase}")
    
    # Get all chapters and labs
    print("\nFetching list of chapters and labs...")
    all_chapters = get_all_chapters_and_labs()
    
    # Filter chapters based on provided patterns/lists
    chapters = filter_chapters(
        all_chapters,
        chapter_patterns=args.chapter_patterns,
        lab_patterns=args.lab_patterns,
        chapter_list=args.chapters,
        lab_list=args.labs
    )
    
    # Display filter information
    if args.chapter_patterns or args.lab_patterns or args.chapters or args.labs:
        print(f"\nFiltering applied:")
        if args.chapter_patterns:
            print(f"  Chapter patterns: {args.chapter_patterns}")
        if args.lab_patterns:
            print(f"  Lab patterns: {args.lab_patterns}")
        if args.chapters:
            print(f"  Explicit chapters: {args.chapters}")
        if args.labs:
            print(f"  Explicit labs: {args.labs}")
        print(f"\nFound {len(all_chapters)} total chapters/labs")
        print(f"Filtered to {len(chapters)} chapters/labs to process:")
    else:
        print(f"\nFound {len(chapters)} chapters/labs to process:")
    
    for i, ch in enumerate(chapters, 1):
        print(f"  {i:3d}. {ch}")
    
    if len(chapters) == 0:
        print("\nERROR: No chapters/labs match the specified filters!")
        sys.exit(1)
    
    # Create log file
    log_dir = Path("artifacts")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"sequential_benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    print(f"\nLog file: {log_file_path}")
    print(f"\n{'='*80}")
    print("TO MONITOR PROGRESS IN ANOTHER TERMINAL, RUN:")
    print(f"tail -f {log_file_path}")
    print(f"{'='*80}\n")
    
    # Track results
    results = {
        "start_time": datetime.now().isoformat(),
        "chapters": [],
        "successful": 0,
        "failed": 0,
        "skipped": 0
    }
    
    with open(log_file_path, 'w') as log_file:
        log_file.write("Sequential Benchmark Runner\n")
        log_file.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Profile: {args.profile}\n")
        if args.update_expectations:
            log_file.write("Mode: --update-expectations enabled\n")
        if args.accept_regressions:
            log_file.write("Mode: --accept-regressions enabled\n")
        if args.cold_start:
            log_file.write("Mode: --cold-start enabled\n")
        if args.reproducible:
            log_file.write("Mode: --reproducible enabled\n")
        if args.suite_timeout is not None:
            log_file.write(f"Suite timeout: {args.suite_timeout} seconds\n")
        if args.timeout_multiplier is not None:
            log_file.write(f"Timeout multiplier: {args.timeout_multiplier}x\n")
        if args.skip_verify:
            log_file.write("Mode: --skip-verify enabled\n")
        if args.verify_phase:
            log_file.write(f"Verify phase: {args.verify_phase}\n")
        log_file.write(f"Total chapters/labs: {len(chapters)}\n")
        log_file.write(f"Chapters/labs list:\n")
        for ch in chapters:
            log_file.write(f"  - {ch}\n")
        log_file.write("\n")
        log_file.flush()
        
        # Process each chapter/lab sequentially
        for idx, chapter in enumerate(chapters, 1):
            print(f"\n\nProcessing {idx}/{len(chapters)}: {chapter}")
            
            chapter_start_time = time.time()
            success = run_benchmark_for_chapter(
                chapter, 
                log_file,
                update_expectations=args.update_expectations,
                accept_regressions=args.accept_regressions,
                cold_start=args.cold_start,
                reproducible=args.reproducible,
                suite_timeout=args.suite_timeout,
                timeout_multiplier=args.timeout_multiplier,
                profile=args.profile,
                kill_gpu=kill_gpu,
                skip_verify=args.skip_verify,
                verify_phase=args.verify_phase
            )
            chapter_duration = time.time() - chapter_start_time
            
            result_entry = {
                "chapter": chapter,
                "index": idx,
                "success": success,
                "duration_seconds": chapter_duration,
                "timestamp": datetime.now().isoformat()
            }
            results["chapters"].append(result_entry)
            
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
            
            # Write intermediate results
            results_file = log_dir / f"sequential_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Small delay between runs to ensure clean state
            if idx < len(chapters):
                print(f"\nWaiting 5 seconds before next chapter/lab...")
                time.sleep(5)
    
    # Final summary
    results["end_time"] = datetime.now().isoformat()
    total_duration = sum(ch["duration_seconds"] for ch in results["chapters"])
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total chapters/labs processed: {len(chapters)}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes, {total_duration/3600:.2f} hours)")
    print(f"Start time: {results['start_time']}")
    print(f"End time: {results['end_time']}")
    print(f"\nLog file: {log_file_path}")
    print(f"Results JSON: {results_file}")
    print("="*80)
    
    # Write final results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Show how to tail the log file
    print(f"\n{'='*80}")
    print("TO MONITOR PROGRESS, RUN:")
    print(f"tail -f {log_file_path}")
    print(f"{'='*80}\n")
    
    # Exit with error code if any failed
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

