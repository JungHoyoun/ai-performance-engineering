'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { getStatusColor } from '@/lib/utils';

interface StatusChartProps {
  counts: {
    succeeded: number;
    failed: number;
    skipped: number;
  };
  height?: number;
}

export function StatusChart({ counts, height = 240 }: StatusChartProps) {
  const total = counts.succeeded + counts.failed + counts.skipped;
  const data = [
    { name: 'Succeeded', value: counts.succeeded, status: 'succeeded' },
    { name: 'Failed', value: counts.failed, status: 'failed' },
    { name: 'Skipped', value: counts.skipped, status: 'skipped' },
  ].filter((entry) => entry.value > 0);

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-semibold text-white">Status Distribution</h3>
        <span className="badge badge-info">{total} total</span>
      </div>
      <div className="card-body">
        {total === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 text-white/50 text-sm">
            <span>No benchmark results yet.</span>
            <span className="text-xs text-white/30 mt-2">
              Run benchmarks to populate pass/fail status.
            </span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={height}>
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                nameKey="name"
                innerRadius={50}
                outerRadius={80}
                paddingAngle={4}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getStatusColor(entry.status)} />
                ))}
              </Pie>
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload || payload.length === 0) return null;
                  const item = payload[0].payload as any;
                  const percent = total ? ((item.value / total) * 100).toFixed(1) : '0.0';
                  return (
                    <div className="rounded-lg border border-white/10 bg-black/80 px-3 py-2 text-xs text-white/80">
                      <div className="font-semibold text-white mb-1">{item.name}</div>
                      <div className="flex justify-between gap-4">
                        <span>Count</span>
                        <span className="text-white">{item.value}</span>
                      </div>
                      <div className="flex justify-between gap-4">
                        <span>Share</span>
                        <span className="text-white">{percent}%</span>
                      </div>
                    </div>
                  );
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
