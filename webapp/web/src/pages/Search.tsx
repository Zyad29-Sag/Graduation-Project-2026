import { useState, type FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  Image as ImageIcon,
  MessageSquare,
  Search as SearchIcon,
  SlidersHorizontal,
} from "lucide-react";
import { searchFilters, searchImage, searchText } from "../api/client";
import type { SearchResponse } from "../api/types";
import { Card, Empty, PageHeader, Spinner } from "../components/ui";
import { PersonCard } from "../components/PersonCard";
import { PersonDrawer } from "../components/PersonDrawer";

type Tab = "chat" | "image" | "filters";

const TABS: { id: Tab; label: string; icon: typeof MessageSquare }[] = [
  { id: "chat", label: "Chatbot", icon: MessageSquare },
  { id: "image", label: "By image", icon: ImageIcon },
  { id: "filters", label: "Filters", icon: SlidersHorizontal },
];

function clean(f: Record<string, string>) {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(f))
    if (v && v.trim()) out[k] = k === "camera" ? Number(v) : v.trim();
  return out;
}

function Results({ res, onPick }: { res?: SearchResponse; onPick: (id: string) => void }) {
  if (!res) return null;
  return (
    <div className="mt-5">
      {res.note && (
        <div className="mb-3 rounded-lg border border-amber-400/30 bg-amber-400/5 px-3 py-2 text-xs text-amber-200/80">
          {res.note}
        </div>
      )}
      {res.results.length === 0 ? (
        <Empty>No matches.</Empty>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {res.results.map((p) => (
            <PersonCard key={p.person_id} p={p} onClick={() => onPick(p.person_id)} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function SearchPage() {
  const [tab, setTab] = useState<Tab>("chat");
  const [sel, setSel] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const mText = useMutation({ mutationFn: () => searchText(query, 12) });

  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"body" | "face">("body");
  const mImg = useMutation({ mutationFn: () => searchImage(file!, mode, 8) });

  const [filters, setFilters] = useState<Record<string, string>>({});
  const setF = (k: string, v: string) => setFilters((p) => ({ ...p, [k]: v }));
  const mFilt = useMutation({ mutationFn: () => searchFilters(clean(filters)) });

  const active = tab === "chat" ? mText : tab === "image" ? mImg : mFilt;

  const submitChat = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) mText.mutate();
  };

  return (
    <>
      <PageHeader title="Search" subtitle="Find a person three ways" />

      <div className="mb-5 flex gap-2">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`btn ${tab === id ? "btn-primary" : "btn-ghost"}`}
          >
            <Icon size={16} /> {label}
          </button>
        ))}
      </div>

      {tab === "chat" && (
        <Card className="p-4">
          <form onSubmit={submitChat} className="flex gap-2">
            <input
              className="input"
              placeholder='Describe the person — e.g. "a man wearing glasses and a dark jacket"'
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button className="btn-primary flex-none" disabled={mText.isPending}>
              {mText.isPending ? <Spinner /> : <SearchIcon size={16} />}
              Search
            </button>
          </form>
          <p className="mt-2 text-xs text-emerald-200/40">
            Semantic search over LLM body descriptions (nearest in meaning, not keywords).
          </p>
        </Card>
      )}

      {tab === "image" && (
        <Card className="p-4">
          <div className="flex flex-wrap items-center gap-3">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="text-sm text-emerald-100 file:mr-3 file:rounded-lg file:border-0 file:bg-emerald file:px-3 file:py-2 file:text-ink-900"
            />
            <div className="flex gap-1">
              {(["body", "face"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`chip ${mode === m ? "border-emerald text-emerald-glow" : ""}`}
                >
                  {m}
                </button>
              ))}
            </div>
            <button
              className="btn-primary"
              disabled={!file || mImg.isPending}
              onClick={() => mImg.mutate()}
            >
              {mImg.isPending ? <Spinner /> : <SearchIcon size={16} />}
              Search
            </button>
          </div>
          <p className="mt-2 text-xs text-emerald-200/40">
            {mode === "body"
              ? "OSNet body Re-ID over the gallery."
              : "InsightFace face match over the isolated face store."}
          </p>
        </Card>
      )}

      {tab === "filters" && (
        <Card className="space-y-3 p-4">
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            {[
              ["gender", "Male"],
              ["age_range", "30-45"],
              ["ethnicity", "White"],
              ["glasses", "Glasses"],
              ["clothing_top_color", "black"],
              ["clothing_bottom_color", "blue"],
              ["hair_color", "black"],
              ["camera", "cam id"],
            ].map(([k, ph]) => (
              <input
                key={k}
                className="input"
                placeholder={`${k} (${ph})`}
                value={filters[k] ?? ""}
                onChange={(e) => setF(k, e.target.value)}
              />
            ))}
          </div>
          <button className="btn-primary" disabled={mFilt.isPending} onClick={() => mFilt.mutate()}>
            {mFilt.isPending ? <Spinner /> : <SearchIcon size={16} />}
            Apply filters
          </button>
          <p className="text-xs text-emerald-200/40">
            Face attributes (gender / age / ethnicity / glasses) work now; clothing/hair filters need
            descriptions (run --describe-all).
          </p>
        </Card>
      )}

      <Results res={active.data as SearchResponse | undefined} onPick={setSel} />
      <PersonDrawer personId={sel} onClose={() => setSel(null)} />
    </>
  );
}
