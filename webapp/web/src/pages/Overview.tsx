import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getStats, listPersons } from "../api/client";
import { Card, PageHeader, Spinner, StatCard } from "../components/ui";
import { PersonCard } from "../components/PersonCard";
import { PersonDrawer } from "../components/PersonDrawer";

function Dist({ title, data }: { title: string; data: Record<string, number> }) {
  const total = Object.values(data).reduce((a, b) => a + b, 0) || 1;
  return (
    <Card className="p-4">
      <div className="mb-3 text-xs uppercase tracking-wider text-emerald-200/50">{title}</div>
      <div className="space-y-2">
        {Object.entries(data).length === 0 && <div className="text-xs text-emerald-200/30">—</div>}
        {Object.entries(data).map(([k, v]) => (
          <div key={k}>
            <div className="flex justify-between text-xs text-emerald-100/70">
              <span>{k}</span>
              <span>{v}</span>
            </div>
            <div className="mt-1 h-1.5 rounded-full bg-ink-600">
              <div
                className="h-full rounded-full bg-emerald"
                style={{ width: `${(v / total) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

export default function Overview() {
  const [sel, setSel] = useState<string | null>(null);
  const stats = useQuery({ queryKey: ["stats"], queryFn: getStats });
  const recent = useQuery({ queryKey: ["persons", { recent: 8 }], queryFn: () => listPersons({ limit: 8 }) });

  if (stats.isLoading)
    return (
      <div className="grid h-64 place-items-center">
        <Spinner className="text-emerald" />
      </div>
    );
  const s = stats.data!;

  return (
    <>
      <PageHeader title="Overview" subtitle="System-wide identity & activity snapshot" />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Tracked people" value={s.persons} sub={`${s.multi_camera} multi-camera`} />
        <StatCard label="Body embeddings" value={s.total_body_embeddings} sub={`across ${s.persons} galleries`} />
        <StatCard label="Described" value={`${s.described}/${s.persons}`} sub={`${s.undescribed} pending`} />
        <StatCard label="Alerts" value={s.alerts} sub="violence / anomaly" />
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-3">
        <Dist title="By status" data={s.by_status} />
        <Dist title="Per-camera sightings" data={s.per_camera_sightings} />
        <Dist title="Ethnicity" data={s.distributions.ethnicity} />
      </div>

      <h2 className="mb-3 mt-8 text-sm font-semibold text-emerald-glow">Recent people</h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {recent.data?.items.map((p) => (
          <PersonCard key={p.person_id} p={p} onClick={() => setSel(p.person_id)} />
        ))}
      </div>

      <PersonDrawer personId={sel} onClose={() => setSel(null)} />
    </>
  );
}
