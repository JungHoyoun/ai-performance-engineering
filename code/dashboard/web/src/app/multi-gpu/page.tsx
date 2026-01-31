import { DashboardShell } from '@/components/DashboardShell';

export default function MultiGpuPage() {
  return (
    <DashboardShell
      title="AI Performance Dashboard"
      subtitle="Topology, NVLink bandwidth, and multi-GPU scaling."
    >
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-white">Multi-GPU Status</h2>
          <span className="badge badge-success">Ready</span>
        </div>
        <div className="card-body space-y-3 text-white/70 text-sm">
          <p>
            This tab will visualize GPU topology, NVLink paths, and communication bottlenecks
            for distributed benchmarks.
          </p>
          <p className="text-white/50">
            Run multi-GPU benchmarks to capture topology and scaling data.
          </p>
        </div>
      </div>
    </DashboardShell>
  );
}
