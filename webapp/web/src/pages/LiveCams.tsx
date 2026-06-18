import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Eye, EyeOff, Info } from "lucide-react";
import { getCameras, streamUrl } from "../api/client";
import { Card, Empty, PageHeader, Spinner } from "../components/ui";

function LegendDot({ className }: { className: string }) {
  return <span className={`inline-block h-2.5 w-2.5 rounded-sm ${className}`} />;
}

export default function LiveCams() {
  const cams = useQuery({ queryKey: ["cameras"], queryFn: getCameras });
  const [overlay, setOverlay] = useState(true);

  if (cams.isLoading)
    return (
      <div className="grid h-64 place-items-center">
        <Spinner className="text-emerald" />
      </div>
    );

  const overlayAvailable = cams.data?.overlay_available ?? false;
  const showOverlay = overlay && overlayAvailable;

  return (
    <>
      <div className="flex items-start justify-between gap-3">
        <PageHeader title="Live Cameras" subtitle="Demo playback of the WiseNet camera set" />
        <button
          onClick={() => setOverlay((v) => !v)}
          disabled={!overlayAvailable}
          className={`btn ${showOverlay ? "btn-primary" : "btn-ghost"} flex-none`}
          title={
            overlayAvailable
              ? "Toggle real detection boxes / IDs"
              : "No overlay data — run `python -m webapp.api.tools.record_overlays`"
          }
        >
          {showOverlay ? <Eye size={16} /> : <EyeOff size={16} />}
          Overlays {showOverlay ? "on" : "off"}
        </button>
      </div>

      {overlayAvailable ? (
        <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-line bg-ink-900/40 px-3 py-2 text-xs text-emerald-100/70">
          <span className="font-medium text-emerald-100/90">Legend</span>
          <span className="flex items-center gap-1.5">
            <LegendDot className="bg-white" /> collecting
          </span>
          <span className="flex items-center gap-1.5">
            <LegendDot className="bg-emerald" /> identified (P:id)
          </span>
          <span className="flex items-center gap-1.5">
            <LegendDot className="bg-cyan-300" /> double-border = returning
          </span>
          <span className="text-emerald-200/40">colors are stable per person across cameras</span>
        </div>
      ) : (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-xs text-amber-200/70">
          <Info size={15} className="mt-0.5 flex-none text-amber-300" />
          <span>
            Raw playback. To show real detection boxes/IDs, run{" "}
            <code className="rounded bg-ink-900 px-1">python -m webapp.api.tools.record_overlays</code>{" "}
            once, then reload.
          </span>
        </div>
      )}

      {cams.data?.cameras.length === 0 ? (
        <Empty>No camera videos found.</Empty>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {cams.data?.cameras.map((c) => (
            <Card key={c.cam_id} className="overflow-hidden">
              <div className="flex items-center justify-between border-b border-line px-3 py-2 text-sm">
                <span className="text-emerald-100">{c.name}</span>
                {c.overlap_group && (
                  <span className="chip text-cyan-300">overlap {c.overlap_group.join("↔")}</span>
                )}
              </div>
              <div className="aspect-video bg-ink-900">
                {c.available ? (
                  <img
                    key={`${c.cam_id}-${showOverlay}`}
                    src={streamUrl(c.cam_id, showOverlay && (c.overlay_available ?? false))}
                    className="h-full w-full object-cover"
                    alt={c.name}
                  />
                ) : (
                  <div className="grid h-full place-items-center text-xs text-emerald-200/30">
                    video unavailable
                  </div>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
