export interface Benchmark {
  name: string;
  chapter: string;
  type: string;
  status: 'succeeded' | 'failed' | 'skipped';
  baseline_time_ms: number;
  optimized_time_ms: number;
  speedup: number;
  raw_speedup?: number;
  speedup_capped?: boolean;
  optimization_goal?: 'speed' | 'memory';
  baseline_memory_mb?: number | null;
  optimized_memory_mb?: number | null;
  memory_savings_pct?: number | null;
  p75_ms?: number | null;
  error?: string;
}

export interface BenchmarkSummary {
  total: number;
  succeeded: number;
  failed: number;
  skipped: number;
  avg_speedup: number;
  max_speedup: number;
  min_speedup: number;
}

export interface BenchmarkPagination {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export interface BenchmarkFilters {
  search?: string | null;
  status?: string[];
  chapter?: string[];
  benchmark?: string | null;
  optimization_goal?: string | null;
  sort_field?: string;
  sort_dir?: string;
}

export interface BenchmarkOverview {
  timestamp?: string;
  summary: BenchmarkSummary;
  status_counts: {
    succeeded: number;
    failed: number;
    skipped: number;
  };
  top_speedups: Benchmark[];
  chapter_stats: Array<{
    chapter: string;
    count: number;
    succeeded: number;
    avg_speedup: number;
    max_speedup: number;
  }>;
}

export interface BenchmarkPage {
  timestamp?: string;
  summary: BenchmarkSummary;
  benchmarks: Benchmark[];
  pagination: BenchmarkPagination;
  filters: BenchmarkFilters;
}

export interface BenchmarkData {
  benchmarks: Benchmark[];
  summary: BenchmarkSummary;
  timestamp: string;
  speedup_cap?: number;
}

export interface GpuInfo {
  name: string;
  memory_total: number;      // MB
  memory_used: number;       // MB
  utilization: number;
  temperature: number;
  temperature_hbm?: number | null;
  power?: number;            // watts
  power_draw?: number;       // legacy key
  power_limit?: number;      // watts
  compute_capability?: string;
  driver_version?: string;
  cuda_version?: string;
  clock_graphics?: number;
  clock_memory?: number;
  fan_speed?: number | null;
  pstate?: string | null;
  live?: boolean;
}

export interface SoftwareInfo {
  python_version: string;
  pytorch_version: string;
  cuda_version: string;
  cudnn_version: string;
  triton_version?: string;
}

export interface LLMAnalysis {
  summary: string;
  key_findings: string[];
  recommendations: string[];
  bottlenecks: Array<{
    name: string;
    severity: 'high' | 'medium' | 'low';
    description: string;
    recommendation: string;
  }>;
}

export interface ProfilerData {
  kernels: Array<{
    name: string;
    duration_ms: number;
    memory_mb: number;
    occupancy: number;
  }>;
  memory_timeline: Array<{
    timestamp: number;
    allocated_mb: number;
    reserved_mb: number;
  }>;
  flame_data?: unknown;
}

export interface BenchmarkRunSummary {
  date: string;
  timestamp: string;
  benchmark_count: number;
  avg_speedup: number;
  max_speedup: number;
  successful: number;
  failed: number;
  source: string;
}

export interface BenchmarkHistory {
  total_runs: number;
  latest?: string | null;
  runs: BenchmarkRunSummary[];
}

export interface BenchmarkTrendPoint {
  date: string;
  avg_speedup: number;
  max_speedup: number;
  benchmark_count: number;
}

export interface BenchmarkTrends {
  history: BenchmarkTrendPoint[];
  by_date: BenchmarkTrendPoint[];
  avg_speedup: number;
  run_count: number;
  best_ever?: {
    date?: string;
    speedup?: number;
  };
  improvements?: Array<{
    date: string;
    delta: number;
  }>;
  regressions?: Array<{
    date: string;
    delta: number;
  }>;
}

export interface BenchmarkCompareDelta {
  key: string;
  chapter: string;
  name: string;
  baseline_speedup: number;
  candidate_speedup: number;
  delta: number;
  delta_pct?: number | null;
  baseline_status: string;
  candidate_status: string;
  status_changed: boolean;
  baseline_time_ms?: number | null;
  candidate_time_ms?: number | null;
  baseline_optimized_time_ms?: number | null;
  candidate_optimized_time_ms?: number | null;
}

export interface BenchmarkCompareRun {
  path: string;
  timestamp?: string | null;
  summary: BenchmarkSummary;
}

export interface BenchmarkCompareBenchmark {
  key: string;
  chapter: string;
  name: string;
  status: string;
  speedup: number;
  baseline_time_ms?: number | null;
  optimized_time_ms?: number | null;
  optimization_goal?: string | null;
}

export interface BenchmarkCompareResult {
  baseline: BenchmarkCompareRun;
  candidate: BenchmarkCompareRun;
  overlap: {
    common: number;
    added: number;
    removed: number;
    baseline_total: number;
    candidate_total: number;
  };
  deltas: BenchmarkCompareDelta[];
  regressions: BenchmarkCompareDelta[];
  improvements: BenchmarkCompareDelta[];
  added_benchmarks: BenchmarkCompareBenchmark[];
  removed_benchmarks: BenchmarkCompareBenchmark[];
  status_transitions: Record<string, number>;
}

export interface Tab {
  id: string;
  label: string;
  icon: string;
}
