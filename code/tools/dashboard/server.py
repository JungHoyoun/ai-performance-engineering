#!/usr/bin/env python3
"""
GPU Performance Lab Dashboard Server

A sleek web dashboard for viewing benchmark results, LLM analysis,
and optimization insights.

Usage:
    python -m tools.dashboard.server [--port 8080] [--data results.json]
"""

import argparse
import http.server
import json
import os
import socketserver
import webbrowser
from pathlib import Path
from typing import Optional
import threading
import time
import glob


# Find the code root (3 levels up from this file)
CODE_ROOT = Path(__file__).parent.parent.parent


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves the dashboard and API endpoints."""
    
    def __init__(self, *args, data_file: Optional[Path] = None, **kwargs):
        self.data_file = data_file
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/api/data':
            self.send_json_response(self.load_benchmark_data())
        elif self.path == '/api/gpu':
            self.send_json_response(self.get_gpu_info())
        elif self.path == '/api/llm-analysis':
            self.send_json_response(self.load_llm_analysis())
        elif self.path == '/api/profiles':
            self.send_json_response(self.load_profile_data())
        elif self.path.startswith('/api/'):
            self.send_json_response({"error": "Unknown API endpoint"})
        else:
            super().do_GET()
    
    def send_json_response(self, data: dict):
        """Send a JSON response."""
        response = json.dumps(data, default=str).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)
    
    def load_benchmark_data(self) -> dict:
        """Load benchmark data from JSON file."""
        # Try explicit data file first
        if self.data_file and self.data_file.exists():
            with open(self.data_file) as f:
                return self._transform_benchmark_data(json.load(f))
        
        # Try default location
        default_path = CODE_ROOT / 'benchmark_test_results.json'
        if default_path.exists():
            with open(default_path) as f:
                return self._transform_benchmark_data(json.load(f))
        
        # Return empty structure
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmarks": [],
            "summary": {
                "total_benchmarks": 0,
                "avg_speedup": 0,
                "max_speedup": 0,
            }
        }
    
    def _transform_benchmark_data(self, raw_data: dict) -> dict:
        """Transform raw benchmark data to dashboard format."""
        benchmarks = []
        all_speedups = []
        
        for chapter_result in raw_data.get('results', []):
            chapter = chapter_result.get('chapter', 'unknown')
            
            for bench in chapter_result.get('benchmarks', []):
                name = bench.get('example', 'unknown')
                baseline_time = bench.get('baseline_time_ms', 0)
                best_speedup = bench.get('best_speedup', 1.0)
                status = bench.get('status', 'unknown')
                bench_type = bench.get('type', 'python')
                
                # Get best optimized time
                optimized_time = baseline_time / best_speedup if best_speedup > 0 else baseline_time
                
                # Extract GPU metrics if available
                gpu_metrics = bench.get('baseline_gpu_metrics', {})
                
                # Extract optimization details
                optimizations = []
                for opt in bench.get('optimizations', []):
                    optimizations.append({
                        'technique': opt.get('technique', ''),
                        'speedup': opt.get('speedup', 1.0),
                        'time_ms': opt.get('time_ms', 0),
                        'file': opt.get('file', '')
                    })
                
                benchmarks.append({
                    'name': name,
                    'chapter': chapter,
                    'type': bench_type,
                    'baseline_time_ms': baseline_time,
                    'optimized_time_ms': optimized_time,
                    'speedup': best_speedup,
                    'status': status,
                    'gpu_temp': gpu_metrics.get('temperature_gpu_c'),
                    'gpu_power': gpu_metrics.get('power_draw_w'),
                    'gpu_util': gpu_metrics.get('utilization_gpu_pct'),
                    'optimizations': optimizations,
                    'error': bench.get('error'),
                    'p75_ms': bench.get('baseline_p75_ms'),
                })
                
                if best_speedup > 0:
                    all_speedups.append(best_speedup)
        
        # Sort by speedup descending
        benchmarks.sort(key=lambda x: x['speedup'], reverse=True)
        
        # Calculate summary
        summary = raw_data.get('results', [{}])[0].get('summary', {}) if raw_data.get('results') else {}
        
        return {
            "timestamp": raw_data.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S")),
            "benchmarks": benchmarks,
            "summary": {
                "total_benchmarks": len(benchmarks),
                "avg_speedup": sum(all_speedups) / len(all_speedups) if all_speedups else 0,
                "max_speedup": max(all_speedups) if all_speedups else 0,
                "min_speedup": min(all_speedups) if all_speedups else 0,
                "successful": summary.get('successful', 0),
                "failed": summary.get('failed', 0),
                "failed_regression": summary.get('failed_regression', 0),
            }
        }
    
    def load_llm_analysis(self) -> dict:
        """Load LLM analysis files from benchmark_profiles."""
        analysis = []
        
        profiles_dir = CODE_ROOT / 'benchmark_profiles'
        if profiles_dir.exists():
            for md_file in profiles_dir.rglob('llm_analysis*.md'):
                try:
                    content = md_file.read_text()
                    # Extract chapter from path
                    parts = md_file.relative_to(profiles_dir).parts
                    chapter = parts[0] if parts else 'unknown'
                    # Extract benchmark name from filename
                    name = md_file.stem.replace('llm_analysis_', '')
                    
                    analysis.append({
                        'chapter': chapter,
                        'name': name,
                        'content': content,
                        'path': str(md_file.relative_to(CODE_ROOT)),
                    })
                except Exception as e:
                    print(f"Error loading {md_file}: {e}")
        
        # Also look for explanation files
        for md_file in CODE_ROOT.rglob('*_llm_explanation.md'):
            try:
                content = md_file.read_text()
                analysis.append({
                    'chapter': md_file.parent.name,
                    'name': md_file.stem.replace('_llm_explanation', ''),
                    'content': content,
                    'path': str(md_file.relative_to(CODE_ROOT)),
                    'type': 'explanation'
                })
            except Exception as e:
                print(f"Error loading {md_file}: {e}")
        
        return {"analyses": analysis, "count": len(analysis)}
    
    def load_profile_data(self) -> dict:
        """Load available profile data."""
        profiles = []
        
        profiles_dir = CODE_ROOT / 'benchmark_profiles'
        if profiles_dir.exists():
            for chapter_dir in profiles_dir.iterdir():
                if chapter_dir.is_dir():
                    chapter = chapter_dir.name
                    chapter_profiles = {
                        'chapter': chapter,
                        'nsys_reports': [],
                        'ncu_reports': [],
                        'torch_traces': [],
                    }
                    
                    for f in chapter_dir.iterdir():
                        if f.suffix == '.nsys-rep':
                            chapter_profiles['nsys_reports'].append(f.name)
                        elif f.suffix == '.ncu-rep':
                            chapter_profiles['ncu_reports'].append(f.name)
                        elif f.suffix == '.json' and 'torch_trace' in f.name:
                            chapter_profiles['torch_traces'].append(f.name)
                    
                    if any([chapter_profiles['nsys_reports'], 
                            chapter_profiles['ncu_reports'],
                            chapter_profiles['torch_traces']]):
                        profiles.append(chapter_profiles)
        
        return {"profiles": profiles}
    
    def get_gpu_info(self) -> dict:
        """Get GPU information using nvidia-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                return {
                    "name": parts[0],
                    "temperature": float(parts[1]),
                    "power": float(parts[2]),
                    "memory_used": float(parts[3]),
                    "memory_total": float(parts[4]),
                    "utilization": float(parts[5]),
                    "live": True
                }
        except Exception:
            pass
        
        return {
            "name": "NVIDIA B200",
            "temperature": 42,
            "power": 192,
            "memory_used": 1024,
            "memory_total": 196608,
            "utilization": 0,
            "live": False
        }
    
    def log_message(self, format, *args):
        """Suppress logging for cleaner output."""
        pass


def create_handler(data_file: Optional[Path] = None):
    """Create a handler class with the data file bound."""
    def handler(*args, **kwargs):
        return DashboardHandler(*args, data_file=data_file, **kwargs)
    return handler


def serve_dashboard(port: int = 8080, data_file: Optional[Path] = None, open_browser: bool = True):
    """Start the dashboard server."""
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    handler = create_handler(data_file)
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ⚡ GPU Performance Lab Dashboard                               ║
║                                                                  ║
║   Server running at: {url:<40} ║
║   Data source: {str(data_file or 'benchmark_test_results.json')[:40]:<40} ║
║                                                                  ║
║   API Endpoints:                                                 ║
║   • GET /api/data         - Benchmark results                    ║
║   • GET /api/gpu          - Live GPU status                      ║
║   • GET /api/llm-analysis - LLM analysis reports                 ║
║   • GET /api/profiles     - Available profile data               ║
║                                                                  ║
║   Press Ctrl+C to stop                                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        
        if open_browser:
            # Open browser after a short delay
            def open_delayed():
                time.sleep(0.5)
                webbrowser.open(url)
            threading.Thread(target=open_delayed, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard server stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="GPU Performance Lab Dashboard Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.dashboard.server
  python -m tools.dashboard.server --port 3000
  python -m tools.dashboard.server --data artifacts/benchmark_test_results.json
        """
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8080,
        help='Port to run the server on (default: 8080)'
    )
    parser.add_argument(
        '--data', '-d',
        type=Path,
        default=None,
        help='Path to benchmark results JSON file'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    args = parser.parse_args()
    serve_dashboard(port=args.port, data_file=args.data, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
