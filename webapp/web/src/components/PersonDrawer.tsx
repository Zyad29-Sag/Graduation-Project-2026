import { useEffect, useState, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { GitMerge, RotateCw, Scissors, Trash2, X } from "lucide-react";
import {
  deletePerson,
  editAttributes,
  getPerson,
  mergePersons,
  redescribe,
  splitPerson,
} from "../api/client";
import { useAuth } from "../auth/AuthContext";
import type { PersonDetail } from "../api/types";
import { Spinner, StatusBadge } from "./ui";
import { Thumb } from "./Thumb";

const ATTR_FIELDS = ["name", "gender", "age_range", "ethnicity", "glasses"] as const;

export function PersonDrawer({
  personId,
  onClose,
}: {
  personId: string | null;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const { user } = useAuth();
  const canWrite = user?.role === "admin" || user?.role === "operator";

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["person", personId],
    queryFn: () => getPerson(personId!),
    enabled: !!personId,
  });

  const [attrs, setAttrs] = useState<Record<string, string>>({});
  const [splitSel, setSplitSel] = useState<number[]>([]);
  const [mergeId, setMergeId] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    if (data) {
      setAttrs(Object.fromEntries(ATTR_FIELDS.map((f) => [f, (data as any)[f] ?? ""])));
      setSplitSel([]);
      setMergeId("");
      setMsg(null);
    }
  }, [data]);

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["person", personId] });
    qc.invalidateQueries({ queryKey: ["persons"] });
    qc.invalidateQueries({ queryKey: ["stats"] });
    qc.invalidateQueries({ queryKey: ["audit"] });
  };

  const mSave = useMutation({
    mutationFn: () =>
      editAttributes(
        personId!,
        Object.fromEntries(Object.entries(attrs).filter(([, v]) => v !== ""))
      ),
    onSuccess: () => {
      setMsg("Attributes saved.");
      invalidate();
    },
  });
  const mRedescribe = useMutation({
    mutationFn: () => redescribe(personId!),
    onSuccess: () => setMsg("Re-describe queued (needs Ollama to process)."),
  });
  const mSplit = useMutation({
    mutationFn: () => splitPerson(personId!, splitSel),
    onSuccess: (r: any) => {
      setMsg(`Split → new person ${String(r.new_person_id).slice(0, 8)}.`);
      invalidate();
    },
  });
  const mMerge = useMutation({
    mutationFn: () => mergePersons(personId!, mergeId.trim()),
    onSuccess: () => {
      setMsg("Merged.");
      invalidate();
      onClose();
    },
    onError: () => setMsg("Merge failed — check the other person id."),
  });
  const mDelete = useMutation({
    mutationFn: () => deletePerson(personId!),
    onSuccess: () => {
      invalidate();
      onClose();
    },
  });

  if (!personId) return null;
  const d = data as PersonDetail | undefined;

  return (
    <div className="fixed inset-0 z-40 flex">
      <div className="flex-1 bg-black/60" onClick={onClose} />
      <div className="w-full max-w-xl overflow-y-auto border-l border-line bg-ink-800 shadow-2xl">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line bg-ink-800/95 px-5 py-3 backdrop-blur">
          <div className="flex items-center gap-3">
            <span className="font-mono text-sm text-emerald-glow">{personId.slice(0, 8)}</span>
            {d && <StatusBadge status={d.status} />}
          </div>
          <button onClick={onClose} className="text-emerald-200/60 hover:text-emerald-50">
            <X size={20} />
          </button>
        </div>

        {isLoading ? (
          <div className="grid h-64 place-items-center">
            <Spinner className="text-emerald" />
          </div>
        ) : isError || !d ? (
          <div className="grid h-64 place-items-center gap-3 p-6 text-center">
            <p className="text-sm text-red-300">Failed to load this person.</p>
            <button className="btn-ghost" onClick={() => refetch()}>
              Retry
            </button>
          </div>
        ) : (
          <div className="space-y-6 p-5">
            {/* snapshots */}
            <div className="flex gap-2 overflow-x-auto pb-1">
              {d.snapshots.length === 0 && (
                <div className="text-xs text-emerald-200/40">No snapshots.</div>
              )}
              {d.snapshots.map((s) => (
                <Thumb key={s} url={s} className="h-40 w-28 flex-none rounded-lg border border-line" />
              ))}
            </div>

            {/* meta grid */}
            <div className="grid grid-cols-2 gap-3 text-sm">
              <Meta label="Cameras" value={d.cameras.join(", ") || "—"} />
              <Meta label="Gallery" value={`${d.gallery.count} (dim ${d.gallery.dim ?? "?"})`} />
              <Meta label="First seen" value={`cam ${d.first_seen_cam} · ${fmt(d.first_seen_time)}`} />
              <Meta label="Last seen" value={`cam ${d.last_seen_cam} · ${fmt(d.last_seen_time)}`} />
            </div>

            {/* description */}
            <section>
              <H>Description</H>
              {d.description?.summary ? (
                <p className="text-sm text-emerald-100/80">{d.description.summary}</p>
              ) : (
                <p className="text-sm text-emerald-200/40">
                  Not described yet (run --describe-all / re-describe).
                </p>
              )}
            </section>

            {/* journey */}
            <section>
              <H>Cross-camera journey</H>
              <ol className="relative ml-2 border-l border-line pl-4">
                {d.journey.stops.map((s) => (
                  <li key={s.id} className="mb-3">
                    <span className="absolute -left-[5px] mt-1 h-2 w-2 rounded-full bg-emerald" />
                    <div className="text-sm text-emerald-100">
                      Camera {s.cam_id} <span className="text-emerald-200/40">· track {s.track_id}</span>
                    </div>
                    <div className="text-xs text-emerald-200/40">
                      {fmt(s.first_seen)} → {fmt(s.last_seen)}
                    </div>
                  </li>
                ))}
              </ol>
            </section>

            {msg && (
              <div className="rounded-lg border border-emerald/30 bg-emerald/10 px-3 py-2 text-sm text-emerald-glow">
                {msg}
              </div>
            )}

            {/* corrections */}
            {canWrite ? (
              <section className="space-y-5 rounded-xl border border-line bg-ink-700/50 p-4">
                <H>Corrections</H>

                {/* edit attributes */}
                <div>
                  <div className="mb-2 text-xs uppercase tracking-wider text-emerald-200/40">
                    Edit classification
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {ATTR_FIELDS.map((f) => (
                      <input
                        key={f}
                        className="input"
                        placeholder={f}
                        value={attrs[f] ?? ""}
                        onChange={(e) => setAttrs({ ...attrs, [f]: e.target.value })}
                      />
                    ))}
                  </div>
                  <button
                    className="btn-primary mt-2"
                    disabled={mSave.isPending}
                    onClick={() => mSave.mutate()}
                  >
                    {mSave.isPending ? <Spinner /> : "Save attributes"}
                  </button>
                  <button
                    className="btn-ghost ml-2 mt-2"
                    disabled={mRedescribe.isPending}
                    onClick={() => mRedescribe.mutate()}
                  >
                    <RotateCw size={15} /> Re-describe
                  </button>
                </div>

                {/* split */}
                <div>
                  <div className="mb-2 text-xs uppercase tracking-wider text-emerald-200/40">
                    Split — peel selected embeddings into a new person
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {d.gallery.entries.map((e) => {
                      const on = splitSel.includes(e.id);
                      return (
                        <button
                          key={e.id}
                          onClick={() =>
                            setSplitSel(on ? splitSel.filter((x) => x !== e.id) : [...splitSel, e.id])
                          }
                          className={`chip ${on ? "border-emerald text-emerald-glow" : ""}`}
                          title={`${e.angle_tag} · cam ${e.source_cam}`}
                        >
                          #{e.id} {e.angle_tag}
                        </button>
                      );
                    })}
                  </div>
                  <button
                    className="btn-ghost mt-2"
                    disabled={splitSel.length === 0 || mSplit.isPending}
                    onClick={() => mSplit.mutate()}
                  >
                    <Scissors size={15} /> Split {splitSel.length} selected
                  </button>
                </div>

                {/* merge */}
                <div>
                  <div className="mb-2 text-xs uppercase tracking-wider text-emerald-200/40">
                    Merge — fold another person INTO this one
                  </div>
                  <div className="flex gap-2">
                    <input
                      className="input"
                      placeholder="other person_id to absorb"
                      value={mergeId}
                      onChange={(e) => setMergeId(e.target.value)}
                    />
                    <button
                      className="btn-ghost flex-none"
                      disabled={!mergeId.trim() || mMerge.isPending}
                      onClick={() => mMerge.mutate()}
                    >
                      <GitMerge size={15} /> Merge
                    </button>
                  </div>
                </div>

                {/* delete */}
                <div className="border-t border-line pt-4">
                  <button
                    className="btn-danger"
                    disabled={mDelete.isPending}
                    onClick={() => {
                      if (confirm("Delete this person and all their data? This cannot be undone."))
                        mDelete.mutate();
                    }}
                  >
                    <Trash2 size={15} /> Delete person
                  </button>
                </div>
              </section>
            ) : (
              <p className="text-xs text-emerald-200/40">
                Sign in as operator/admin to edit, merge, split or delete.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function H({ children }: { children: ReactNode }) {
  return <h3 className="mb-2 text-sm font-semibold text-emerald-glow">{children}</h3>;
}
function Meta({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded-lg border border-line bg-ink-800 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-emerald-200/40">{label}</div>
      <div className="text-emerald-100">{value}</div>
    </div>
  );
}
function fmt(t?: string | null) {
  if (!t) return "—";
  const d = new Date(t);
  return isNaN(d.getTime()) ? t : d.toLocaleTimeString();
}
