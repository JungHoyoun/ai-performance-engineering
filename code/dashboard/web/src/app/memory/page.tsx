import { DashboardShell } from '@/components/DashboardShell';

export default function MemoryPage() {
  return (
    <DashboardShell
      title="AI Performance Dashboard"
      subtitle="Memory allocation timelines and savings opportunities."
    >
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-white">Memory Analysis</h2>
          <span className="badge badge-info">Coming soon</span>
        </div>
        <div className="card-body space-y-3 text-white/70 text-sm">
          <p>
            Review peak memory, fragmentation, and allocator behavior across benchmark runs.
            Memory-optimized benchmarks will surface here with focused insights.
          </p>
          <p className="text-white/50">
            Run memory-targeted benchmarks to populate this view.
          </p>
        </div>
      </div>
    </DashboardShell>
  );
}
