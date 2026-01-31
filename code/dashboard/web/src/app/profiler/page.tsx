import { DashboardShell } from '@/components/DashboardShell';

export default function ProfilerPage() {
  return (
    <DashboardShell
      title="AI Performance Dashboard"
      subtitle="Kernel timelines, Nsight artifacts, and performance hotspots."
    >
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-white">Profiler Insights</h2>
          <span className="badge badge-warning">Awaiting profiles</span>
        </div>
        <div className="card-body space-y-3 text-white/70 text-sm">
          <p>
            Capture Nsight Systems and Nsight Compute profiles to populate this tab with
            flame graphs, kernel breakdowns, and memory timelines.
          </p>
          <p className="text-white/50">
            Run <span className="font-mono">aisp bench run --profile full</span> to generate artifacts.
          </p>
        </div>
      </div>
    </DashboardShell>
  );
}
