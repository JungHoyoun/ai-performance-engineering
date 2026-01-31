'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, AlertTriangle, Loader2 } from 'lucide-react';
import { getDependencies } from '@/lib/api';

export function DependenciesWidget() {
  const [deps, setDeps] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const data = await getDependencies();
        setDeps(data);
        setError(false);
      } catch (e) {
        setError(true);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm text-white/50">Checking...</span>
      </div>
    );
  }

  if (error) {
    return null; // Hide if unavailable
  }

  const hasIssues = deps?.missing?.length > 0 || deps?.outdated?.length > 0;

  return (
    <div className={`flex items-center gap-3 px-4 py-2 rounded-full border ${
      hasIssues 
        ? 'bg-accent-warning/10 border-accent-warning/20' 
        : 'bg-accent-success/10 border-accent-success/20'
    }`}>
      {hasIssues ? (
        <>
          <AlertTriangle className="w-4 h-4 text-accent-warning" />
          <span className="text-sm text-accent-warning">
            {deps.missing?.length || 0} missing, {deps.outdated?.length || 0} outdated
          </span>
        </>
      ) : (
        <>
          <CheckCircle className="w-4 h-4 text-accent-success" />
          <span className="text-sm text-accent-success">All deps OK</span>
        </>
      )}
    </div>
  );
}
