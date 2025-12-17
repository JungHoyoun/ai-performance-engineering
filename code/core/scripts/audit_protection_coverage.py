#!/usr/bin/env python3
"""
Audit script that PROVES which benchmark protections are actually implemented.

This script inspects the actual codebase to verify that claimed protections
exist and are wired into the benchmark harness.
"""

import ast
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json

# Protection definitions: what function/class/pattern proves it exists
PROTECTIONS = {
    # Timing Protections
    "full_device_sync": {
        "description": "Full device synchronization before timing",
        "evidence": [
            ("core/harness/benchmark_harness.py", r"torch\.cuda\.synchronize\(\)"),
            ("core/harness/benchmark_harness.py", r"full_device_sync"),
        ],
    },
    "stream_auditor": {
        "description": "Audit CUDA streams for unsynchronized work",
        "evidence": [
            ("core/harness/validity_checks.py", r"class StreamAuditor"),
            ("core/harness/validity_checks.py", r"def audit_streams"),
        ],
    },
    "event_timing_cross_validation": {
        "description": "Cross-validate CUDA events with wall clock",
        "evidence": [
            ("core/harness/benchmark_harness.py", r"cross_validate_timing"),
        ],
    },
    "adaptive_iterations": {
        "description": "Dynamically adjust iterations for measurement accuracy",
        "evidence": [
            ("core/harness/benchmark_harness.py", r"adaptive_iterations"),
            ("core/harness/benchmark_harness.py", r"min_total_duration_ms"),
        ],
    },
    "warmup_isolation": {
        "description": "Isolate warmup from measurement (L2 cache clear)",
        "evidence": [
            ("core/harness/benchmark_harness.py", r"isolate_warmup_cache"),
            ("core/harness/l2_cache_utils.py", r"def flush_l2_cache"),
        ],
    },
    
    # Output Protections
    "jitter_check": {
        "description": "Perturb inputs to detect hardcoded outputs",
        "evidence": [
            ("core/benchmark/verify_runner.py", r"def _run_jitter_check"),
            ("core/benchmark/verification.py", r"def select_jitter_dimension"),
        ],
    },
    "fresh_input_check": {
        "description": "Re-run with different seeds to detect caching",
        "evidence": [
            ("core/benchmark/verify_runner.py", r"def _run_fresh_input_check"),
        ],
    },
    "output_tolerance_validation": {
        "description": "Validate outputs within dtype-aware tolerances",
        "evidence": [
            ("core/benchmark/verification.py", r"class ToleranceSpec"),
            ("core/benchmark/verification.py", r"DEFAULT_TOLERANCES"),
            ("core/benchmark/verify_runner.py", r"def _compare_outputs"),
        ],
    },
    "golden_output_cache": {
        "description": "Store reference outputs for comparison",
        "evidence": [
            ("core/benchmark/verify_runner.py", r"class GoldenOutputCache"),
        ],
    },
    "validate_result_method": {
        "description": "Benchmark-provided result validation",
        "evidence": [
            ("core/benchmark/contract.py", r"validate_result"),
        ],
    },
    
    # Workload Protections
    "input_signature_matching": {
        "description": "Verify baseline/optimized have same workload",
        "evidence": [
            ("core/benchmark/verification.py", r"class InputSignature"),
            ("core/benchmark/verify_runner.py", r"_extract_signature"),
        ],
    },
    "workload_invariant_check": {
        "description": "Verify workload metrics match",
        "evidence": [
            ("core/benchmark/verify_runner.py", r"def _extract_workload_metrics"),
            ("core/benchmark/verification.py", r"def compare_workload_metrics"),
        ],
    },
    "config_immutability": {
        "description": "Prevent runtime modification of benchmark config",
        "evidence": [
            ("core/harness/benchmark_harness.py", r"enforce_config_immutability"),
        ],
    },
    
    # Memory Protections
    "memory_allocation_tracker": {
        "description": "Track memory allocations for anomalies",
        "evidence": [
            ("core/harness/validity_checks.py", r"class MemoryAllocationTracker"),
            ("core/harness/validity_checks.py", r"def track_memory_allocations"),
        ],
    },
    "input_output_aliasing_check": {
        "description": "Detect if output aliases input",
        "evidence": [
            ("core/harness/validity_checks.py", r"def check_input_output_aliasing"),
        ],
    },
    "memory_pool_reset": {
        "description": "Reset CUDA memory pool between runs",
        "evidence": [
            ("core/harness/validity_checks.py", r"def reset_cuda_memory_pool"),
        ],
    },
    "l2_cache_clearing": {
        "description": "Clear L2 cache between iterations",
        "evidence": [
            ("core/harness/l2_cache_utils.py", r"def flush_l2_cache"),
            ("core/harness/l2_cache_utils.py", r"def detect_l2_cache_size"),
        ],
    },
    
    # CUDA Protections
    "cuda_graph_cheat_detection": {
        "description": "Detect work during graph capture",
        "evidence": [
            ("core/harness/validity_checks.py", r"class GraphCaptureCheatDetector"),
            ("core/harness/validity_checks.py", r"def detect_graph_capture_cheat"),
        ],
    },
    "force_tensor_evaluation": {
        "description": "Force lazy tensors to evaluate",
        "evidence": [
            ("core/harness/validity_checks.py", r"def force_tensor_evaluation"),
        ],
    },
    
    # Compile Protections
    "compile_cache_clear": {
        "description": "Clear torch.compile/dynamo cache",
        "evidence": [
            ("core/harness/validity_checks.py", r"def clear_compile_cache"),
        ],
    },
    
    # Environment Protections
    "gpu_clock_locking": {
        "description": "Lock GPU clocks for consistent measurements",
        "evidence": [
            ("core/harness/benchmark_harness.py", r"lock_gpu_clocks"),
        ],
    },
    "gpu_state_monitoring": {
        "description": "Monitor GPU temperature and power",
        "evidence": [
            ("core/harness/validity_checks.py", r"def capture_gpu_state"),
            ("core/harness/validity_checks.py", r"class GPUState"),
        ],
    },
    "environment_validation": {
        "description": "Validate execution environment",
        "evidence": [
            ("core/harness/validity_checks.py", r"def validate_environment"),
        ],
    },
    "setup_precomputation_detection": {
        "description": "Detect work done in setup()",
        "evidence": [
            ("core/harness/validity_checks.py", r"def check_setup_precomputation"),
        ],
    },
    
    # Statistical Protections
    "gc_disabled_during_timing": {
        "description": "Disable garbage collection during timing",
        "evidence": [
            ("core/harness/validity_checks.py", r"def gc_disabled"),
        ],
    },
    "stream_sync_check": {
        "description": "Check all CUDA streams are synchronized",
        "evidence": [
            ("core/harness/validity_checks.py", r"def check_stream_sync_completeness"),
        ],
    },
    
    # Distributed Protections
    "distributed_topology_verification": {
        "description": "Verify distributed benchmark topology",
        "evidence": [
            ("core/benchmark/verify_runner.py", r"def verify_distributed"),
            ("core/harness/validity_checks.py", r"def verify_distributed_outputs"),
        ],
    },
    "rank_execution_check": {
        "description": "Verify all ranks execute work",
        "evidence": [
            ("core/harness/validity_checks.py", r"def check_rank_execution"),
        ],
    },
    
    # Quarantine System
    "quarantine_manager": {
        "description": "Track non-compliant benchmarks",
        "evidence": [
            ("core/benchmark/quarantine.py", r"class QuarantineManager"),
        ],
    },
    "quarantine_reasons": {
        "description": "Enumeration of quarantine reasons",
        "evidence": [
            ("core/benchmark/verification.py", r"class QuarantineReason"),
        ],
    },
    
    # Contract Enforcement
    "benchmark_contract": {
        "description": "Enforce benchmark interface requirements",
        "evidence": [
            ("core/benchmark/contract.py", r"class BenchmarkContract"),
            ("core/benchmark/contract.py", r"REQUIRED_METHODS"),
        ],
    },
    "verification_compliance_check": {
        "description": "Check benchmarks implement verification methods",
        "evidence": [
            ("core/benchmark/contract.py", r"check_verification_compliance"),
        ],
    },
}


def check_evidence(code_root: Path, file_path: str, pattern: str) -> Tuple[bool, Optional[int]]:
    """Check if a pattern exists in a file. Returns (found, line_number)."""
    full_path = code_root / file_path
    if not full_path.exists():
        return False, None
    
    try:
        content = full_path.read_text()
        for i, line in enumerate(content.split('\n'), 1):
            if re.search(pattern, line):
                return True, i
        return False, None
    except Exception as e:
        return False, None


def audit_protection(code_root: Path, name: str, info: dict) -> dict:
    """Audit a single protection."""
    result = {
        "name": name,
        "description": info["description"],
        "evidence": [],
        "all_found": True,
    }
    
    for file_path, pattern in info["evidence"]:
        found, line_num = check_evidence(code_root, file_path, pattern)
        evidence_result = {
            "file": file_path,
            "pattern": pattern,
            "found": found,
            "line": line_num,
        }
        result["evidence"].append(evidence_result)
        if not found:
            result["all_found"] = False
    
    return result


def check_harness_integration(code_root: Path) -> Dict[str, bool]:
    """Check that protections are actually wired into the execution path."""
    harness_file = code_root / "core/harness/benchmark_harness.py"
    if not harness_file.exists():
        return {}
    
    content = harness_file.read_text()

    # Quarantine is an orchestration concern (run_benchmarks.py) implemented via VerifyRunner.
    verify_runner_file = code_root / "core/benchmark/verify_runner.py"
    orchestration_file = code_root / "core/harness/run_benchmarks.py"
    verify_runner_content = verify_runner_file.read_text() if verify_runner_file.exists() else ""
    orchestration_content = orchestration_file.read_text() if orchestration_file.exists() else ""
    quarantine_wired = bool(re.search(r"QuarantineManager\s*\(", verify_runner_content))
    quarantine_used = bool(re.search(r"runner\.quarantine\.", orchestration_content))
    
    integrations = {
        "validity_checks_imported": bool(re.search(r"from.*validity_checks import", content)),
        "l2_cache_utils_imported": bool(re.search(r"from.*l2_cache_utils import", content) or 
                                         re.search(r"from.*l2_cache_utils", content)),
        "verify_runner_used": bool(re.search(r"VerifyRunner", content)),
        "quarantine_manager_used": quarantine_wired and quarantine_used,
        "clear_l2_cache_called": bool(re.search(r"clear_l2_cache", content)),
        "gc_disabled_used": bool(re.search(r"gc_disabled", content)),
        "memory_tracker_used": bool(re.search(r"MemoryAllocationTracker|track_memory_allocations", content)),
        "stream_auditor_used": bool(re.search(r"StreamAuditor|audit_streams", content)),
    }
    
    return integrations


def main():
    code_root = Path(__file__).parent.parent.parent
    
    print("=" * 80)
    print("BENCHMARK PROTECTION AUDIT")
    print("=" * 80)
    print(f"Code root: {code_root}")
    print()
    
    # Audit all protections
    results = []
    passed = 0
    failed = 0
    
    for name, info in PROTECTIONS.items():
        result = audit_protection(code_root, name, info)
        results.append(result)
        
        if result["all_found"]:
            passed += 1
            status = "✅ VERIFIED"
        else:
            failed += 1
            status = "❌ MISSING"
        
        print(f"\n{status} {name}")
        print(f"   {info['description']}")
        for ev in result["evidence"]:
            ev_status = "✓" if ev["found"] else "✗"
            line_info = f" (line {ev['line']})" if ev["line"] else ""
            print(f"   {ev_status} {ev['file']}: {ev['pattern']}{line_info}")
    
    # Check harness integration
    print("\n" + "=" * 80)
    print("HARNESS INTEGRATION CHECK")
    print("=" * 80)
    
    integrations = check_harness_integration(code_root)
    for key, found in integrations.items():
        status = "✅" if found else "❌"
        print(f"{status} {key}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Protections verified: {passed}/{len(PROTECTIONS)} ({100*passed/len(PROTECTIONS):.1f}%)")
    print(f"Protections missing:  {failed}/{len(PROTECTIONS)}")
    
    integrations_passed = sum(1 for v in integrations.values() if v)
    print(f"Harness integrations: {integrations_passed}/{len(integrations)}")
    
    # Write detailed JSON report
    report = {
        "summary": {
            "protections_verified": passed,
            "protections_total": len(PROTECTIONS),
            "coverage_percent": round(100*passed/len(PROTECTIONS), 1),
            "harness_integrations": integrations,
        },
        "protections": results,
    }
    
    report_path = code_root / "artifacts" / "protection_audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report: {report_path}")
    
    # Return exit code
    if failed > 0:
        print(f"\n⚠️  {failed} protection(s) not fully verified!")
        return 1
    else:
        print("\n✅ All protections verified!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
