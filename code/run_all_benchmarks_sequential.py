#!/usr/bin/env python3
"""
Script to run full benchmarks with deep profiling on EVERY chapter and lab,
ONE at a time, sequentially.

Requirements:
- NO LLM analysis
- Deep profiling (deep_dive)
- Sequential execution (one chapter/lab at a time)
- NO parallel execution
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

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

def run_benchmark_for_chapter(chapter, log_file):
    """
    Run benchmark with deep profiling for a single chapter/lab.
    
    Args:
        chapter: Chapter/lab name (e.g., 'ch01', 'labs/decode_optimization')
        log_file: File handle to write logs to
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
    
    start_time = time.time()
    
    try:
        # Run benchmark with deep profiling, NO LLM analysis (default is False)
        cmd = [
            "python", "-m", "cli.aisp", "bench", "run",
            "--targets", chapter,
            "--profile", "deep_dive"
            # Note: llm_analysis defaults to False, so we don't need to disable it
        ]
        
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
    print("="*80)
    print("Sequential Benchmark Runner with Deep Profiling")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get all chapters and labs
    print("\nFetching list of chapters and labs...")
    chapters = get_all_chapters_and_labs()
    
    print(f"\nFound {len(chapters)} chapters/labs to process:")
    for i, ch in enumerate(chapters, 1):
        print(f"  {i:3d}. {ch}")
    
    # Create log file
    log_dir = Path("artifacts")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"sequential_benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    print(f"\nLog file: {log_file_path}")
    
    # Track results
    results = {
        "start_time": datetime.now().isoformat(),
        "chapters": [],
        "successful": 0,
        "failed": 0,
        "skipped": 0
    }
    
    with open(log_file_path, 'w') as log_file:
        log_file.write("Sequential Benchmark Runner with Deep Profiling\n")
        log_file.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
            success = run_benchmark_for_chapter(chapter, log_file)
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
    
    # Exit with error code if any failed
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

