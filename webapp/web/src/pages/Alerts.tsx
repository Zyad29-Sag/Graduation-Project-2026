import { useQuery } from "@tanstack/react-query";
import { ShieldAlert } from "lucide-react";
import { getAlerts } from "../api/client";
import { Card, Empty, PageHeader, Spinner } from "../components/ui";

const LEVEL_COLOR: Record<string, string> = {
  VIOLENCE: "text-red-300 border-red-400/40",
  SUSPICIOUS: "text-amber-300 border-amber-400/40",
};

export default function Alerts() {
  const a = useQuery({ queryKey: ["alerts"], queryFn: getAlerts });

  if (a.isLoading)
    return (
      <div className="grid h-64 place-items-center">
        <Spinner className="text-emerald" />
      </div>
    );

  return (
    <>
      <PageHeader title="Alerts" subtitle="Violence & anomaly events" />
      {!a.data || a.data.items.length === 0 ? (
        <Empty>
          <ShieldAlert className="mx-auto mb-3 text-emerald-200/30" size={28} />
          No alerts recorded.
        </Empty>
      ) : (
        <Card className="divide-y divide-line">
          {a.data.items.map((al, i) => (
            <div key={i} className="flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-3">
                <span className={`chip ${LEVEL_COLOR[al.level || ""] || ""}`}>{al.level || "—"}</span>
                <span className="text-sm text-emerald-100">Camera {al.cam_id}</span>
                {typeof al.score === "number" && (
                  <span className="text-xs text-emerald-200/40">score {al.score.toFixed(2)}</span>
                )}
              </div>
              <span className="text-xs text-emerald-200/40">{al.timestamp}</span>
            </div>
          ))}
        </Card>
      )}
    </>
  );
}
