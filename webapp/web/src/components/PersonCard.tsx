import { Thumb } from "./Thumb";
import { StatusBadge } from "./ui";
import type { PersonSummary } from "../api/types";

export function PersonCard({ p, onClick }: { p: PersonSummary; onClick?: () => void }) {
  return (
    <button
      onClick={onClick}
      className="card group overflow-hidden text-left transition hover:border-emerald/50 hover:shadow-glow"
    >
      <div className="relative aspect-[3/4] w-full overflow-hidden bg-ink-800">
        <Thumb url={p.thumbnail_url} className="h-full w-full transition group-hover:scale-105" />
        {typeof p.score === "number" && p.score !== null && (
          <span className="chip absolute right-2 top-2 bg-ink-900/80 text-emerald-glow">
            {(p.score * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <div className="space-y-1 p-3">
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs text-emerald-200/60">{p.person_id.slice(0, 8)}</span>
          <StatusBadge status={p.status} />
        </div>
        <div className="flex flex-wrap gap-1">
          {p.gender && <span className="chip">{p.gender}</span>}
          {p.age_range && <span className="chip">{p.age_range}</span>}
          {p.glasses && p.glasses !== "No Glasses" && <span className="chip">{p.glasses}</span>}
        </div>
        {p.summary && <p className="line-clamp-2 text-xs text-emerald-200/40">{p.summary}</p>}
      </div>
    </button>
  );
}
