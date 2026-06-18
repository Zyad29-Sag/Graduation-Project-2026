import { type ReactNode } from "react";
import { Loader2 } from "lucide-react";

export function Spinner({ className = "" }: { className?: string }) {
  return <Loader2 className={`animate-spin ${className}`} size={18} />;
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`card ${className}`}>{children}</div>;
}

export function StatCard({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <Card className="p-4">
      <div className="text-xs uppercase tracking-wider text-emerald-200/50">{label}</div>
      <div className="mt-1 text-3xl font-semibold text-emerald-glow">{value}</div>
      {sub && <div className="mt-1 text-xs text-emerald-200/40">{sub}</div>}
    </Card>
  );
}

const STATUS_COLORS: Record<string, string> = {
  confirmed: "text-emerald-300 border-emerald-400/40",
  multi_view: "text-cyan-300 border-cyan-400/40",
  unverified: "text-amber-300 border-amber-400/40",
  flagged: "text-red-300 border-red-400/40",
};
export function StatusBadge({ status }: { status?: string | null }) {
  const c = STATUS_COLORS[status || ""] || "text-emerald-100 border-line";
  return <span className={`chip ${c}`}>{status || "—"}</span>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="py-16 text-center text-sm text-emerald-200/40">{children}</div>;
}

export function PageHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: ReactNode }) {
  return (
    <div className="mb-6 flex items-end justify-between">
      <div>
        <h1 className="text-2xl font-semibold text-emerald-50">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-emerald-200/40">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}
