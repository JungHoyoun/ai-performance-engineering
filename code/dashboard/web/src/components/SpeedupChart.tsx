'use client';

import { useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { Benchmark, BenchmarkOverview } from '@/types';
import { formatMs, getSpeedupColor } from '@/lib/utils';

interface SpeedupChartProps {
  benchmarks: Benchmark[];
  chapterStats?: BenchmarkOverview['chapter_stats'];
  height?: number;
  speedupCap?: number;
  onSelect?: (selection: { type: 'benchmark' | 'chapter'; name: string; chapter?: string }) => void;
}

export function SpeedupChart({
  benchmarks,
  chapterStats = [],
  height = 400,
  speedupCap,
  onSelect,
}: SpeedupChartProps) {
  const [view, setView] = useState<'benchmarks' | 'chapters'>('benchmarks');

  const benchmarkData = useMemo(
    () =>
      benchmarks
        .filter((b) => b.status === 'succeeded' && b.speedup)
        .map((b) => ({
          name: b.name.length > 20 ? `${b.name.slice(0, 20)}...` : b.name,
          fullName: b.name,
          speedup: b.speedup,
          rawSpeedup: b.raw_speedup ?? b.speedup,
          capped: b.speedup_capped,
          chapter: b.chapter,
          baseline_time_ms: b.baseline_time_ms,
          optimized_time_ms: b.optimized_time_ms,
        }))
        .sort((a, b) => b.speedup - a.speedup)
        .slice(0, 15),
    [benchmarks]
  );

  const chapterData = useMemo(
    () =>
      chapterStats
        .map((entry) => ({
          name: entry.chapter,
          speedup: entry.avg_speedup,
          max_speedup: entry.max_speedup,
          count: entry.count,
          succeeded: entry.succeeded,
        }))
        .sort((a, b) => b.speedup - a.speedup)
        .slice(0, 12),
    [chapterStats]
  );

  const data = view === 'benchmarks' ? benchmarkData : chapterData;
  const hasCappedValues = benchmarkData.some((d) => d.capped);

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold text-white">
            🚀 {view === 'benchmarks' ? 'Top Speedups' : 'Chapter Speedups'}
          </h3>
          {hasCappedValues && speedupCap && (
            <span className="badge badge-warning">Capped at {speedupCap.toFixed(0)}x</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-white/40">
            <button
              onClick={() => setView('benchmarks')}
              className={`px-2.5 py-1 rounded-full border ${
                view === 'benchmarks'
                  ? 'bg-white/10 border-white/20 text-white'
                  : 'border-white/10 text-white/50 hover:text-white'
              }`}
            >
              Benchmarks
            </button>
            <button
              onClick={() => setView('chapters')}
              className={`px-2.5 py-1 rounded-full border ${
                view === 'chapters'
                  ? 'bg-white/10 border-white/20 text-white'
                  : 'border-white/10 text-white/50 hover:text-white'
              }`}
            >
              Chapters
            </button>
          </div>
          <span className="badge badge-info">{data.length} items</span>
        </div>
      </div>
      <div className="card-body">
        {data.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-white/50 text-sm">
            <span>No speedup data available yet.</span>
            <span className="text-xs text-white/30 mt-2">
              Run benchmarks to populate performance charts.
            </span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={height}>
            <BarChart
              data={data}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                type="number"
                tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                width={100}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload || payload.length === 0) return null;
                  const item = payload[0].payload as any;
                  const baseline =
                    typeof item.baseline_time_ms === 'number' ? formatMs(item.baseline_time_ms) : '—';
                  const optimized =
                    typeof item.optimized_time_ms === 'number' ? formatMs(item.optimized_time_ms) : '—';
                  return (
                    <div className="rounded-lg border border-white/10 bg-black/80 px-3 py-2 text-xs text-white/80">
                      <div className="font-semibold text-white mb-1">
                        {view === 'chapters' ? `Chapter ${label}` : item.fullName}
                      </div>
                      <div className="flex justify-between gap-4">
                        <span>Speedup</span>
                        <span className="text-white">{item.speedup.toFixed(2)}x</span>
                      </div>
                      {view === 'chapters' ? (
                        <>
                          <div className="flex justify-between gap-4">
                            <span>Max speedup</span>
                            <span className="text-white">{item.max_speedup?.toFixed?.(2) ?? '—'}x</span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span>Benchmarks</span>
                            <span className="text-white">{item.succeeded}/{item.count}</span>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="flex justify-between gap-4">
                            <span>Chapter</span>
                            <span className="text-white">{item.chapter}</span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span>Baseline</span>
                            <span className="text-white">{baseline}</span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span>Optimized</span>
                            <span className="text-white">{optimized}</span>
                          </div>
                        </>
                      )}
                    </div>
                  );
                }}
              />
              <ReferenceLine x={1} stroke="rgba(255,255,255,0.3)" strokeDasharray="3 3" />
              <Bar
                dataKey="speedup"
                radius={[0, 4, 4, 0]}
                onClick={(payload) => {
                  if (!onSelect || !payload) return;
                  const item = (payload as any).payload ?? payload;
                  if (view === 'chapters') {
                    onSelect({ type: 'chapter', name: item.name });
                    return;
                  }
                  onSelect({
                    type: 'benchmark',
                    name: item.fullName || item.name,
                    chapter: item.chapter,
                  });
                }}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getSpeedupColor(entry.speedup)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
