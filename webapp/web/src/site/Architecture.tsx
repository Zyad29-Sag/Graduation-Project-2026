import { Link } from "react-router-dom";
import { ArrowRight, Database, GitMerge, Layers, ShieldCheck } from "lucide-react";
import { GlassCard, Reticles, Section, SectionHeading, Tag } from "./ui";

const STAGES = [
  {
    title: "Capture & enhance",
    body: "Each camera runs on its own thread. Dark frames get CLAHE + auto-gamma so detection and embeddings aren't poisoned by shadow.",
    tech: ["Multi-threaded", "CLAHE", "Auto-gamma"],
  },
  {
    title: "Detect & segment",
    body: "YOLOv8n-seg finds people and returns pixel masks; the background is replaced with neutral grey so the embedding encodes the person, not the wall.",
    tech: ["YOLOv8n-seg", "Mask isolation"],
  },
  {
    title: "Track",
    body: "ByteTrack's two-stage IoU association keeps tracks alive through occlusion using low-confidence detections — no identity flicker on turns.",
    tech: ["ByteTrack", "Kalman"],
  },
  {
    title: "Embed",
    body: "A quality gate rejects blurry/tiny/dark crops; survivors feed OSNet. A pose-aware gallery keeps canonical views (front, side, moving) per person.",
    tech: ["OSNet x1.0", "Quality gate", "Pose-aware"],
  },
  {
    title: "Match",
    body: "FAISS does exact nearest-neighbour search. Context-aware thresholds separate same-camera, cross-camera and overlapping-camera cases.",
    tech: ["FAISS IndexFlatIP", "Triple threshold"],
  },
  {
    title: "Reconcile",
    body: "A background worker proposes merges using mean-pool similarity, boosted when two IDs are repeatedly co-visible on overlapping cameras.",
    tech: ["Co-visibility", "Auto-merge"],
  },
  {
    title: "Describe & index",
    body: "A local VLM writes a body description; it's embedded with a sentence model so people are searchable by meaning.",
    tech: ["Qwen2.5-VL", "MiniLM"],
  },
  {
    title: "Face & alerts",
    body: "InsightFace adds names and demographics in an isolated store; a CNN-LSTM daemon raises violence alerts with media.",
    tech: ["InsightFace", "CNN-LSTM"],
  },
];

const PRINCIPLES = [
  {
    icon: Layers,
    title: "Additive integrity",
    body: "Face, demographics and violence are additive signals — cross-camera body identity stays 100% OSNet/ByteTrack-driven. Nothing re-scores an identity behind your back.",
  },
  {
    icon: Database,
    title: "One source of truth",
    body: "SQLite (WAL) holds all metadata and embeddings; FAISS is a fast in-memory mirror that rebuilds from the database and falls back gracefully if absent.",
  },
  {
    icon: ShieldCheck,
    title: "Graceful degradation",
    body: "Every model is flag-gated and auto-disables when its weights are missing. With features off, the system behaves byte-for-byte as before.",
  },
  {
    icon: GitMerge,
    title: "Human-in-the-loop",
    body: "Operators merge, split, edit and re-describe identities — every correction is audit-logged and invalidates the search caches it touches.",
  },
];

export default function Architecture() {
  return (
    <div className="relative">
      {/* Background image merged into the top of the page */}
      <img src="/arch-hud.png" alt="" className="site-merged-img site-merged-img-top h-[700px]" />

      <Section className="relative z-10 pt-16 sm:pt-20">
        <SectionHeading
          eyebrow="// System architecture"
          title="A pipeline built for precision and trust"
          intro="Eight cooperating stages turn raw frames into stable, searchable, accountable identities — engineered to run in real time on CPU."
        />
      </Section>

      {/* Pipeline as a connected vertical flow */}
      <Section className="relative z-10 mt-12">
        <div className="relative">
          <div className="absolute bottom-4 left-[27px] top-4 hidden w-px bg-gradient-to-b from-emerald-400/40 via-emerald-400/15 to-transparent sm:block" />
          <div className="space-y-4">
            {STAGES.map((s, i) => (
              <div key={s.title} className="flex gap-4">
                <div className="relative z-10 hidden h-14 w-14 shrink-0 items-center justify-center rounded-xl border border-emerald-400/30 bg-[#070c10] text-emerald-300 sm:flex">
                  <span className="site-mono text-lg font-bold">{String(i + 1).padStart(2, "0")}</span>
                </div>
                <GlassCard hover className="flex-1 p-5">
                  <div className="flex flex-wrap items-center gap-3">
                    <span className="site-mono text-emerald-400 sm:hidden">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <h3 className="text-lg font-semibold text-white">{s.title}</h3>
                  </div>
                  <p className="mt-2 text-sm leading-relaxed text-slate-400">{s.body}</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {s.tech.map((t) => (
                      <span
                        key={t}
                        className="site-mono rounded-md border border-white/10 bg-white/[0.03] px-2 py-1 text-[11px] text-cyan-200/80"
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                </GlassCard>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Design principles */}
      <Section className="relative z-10 mt-24">
        <SectionHeading eyebrow="// Design principles" title="The rules the system never breaks" />
        <div className="mt-10 grid gap-5 sm:grid-cols-2">
          {PRINCIPLES.map((p) => (
            <GlassCard key={p.title} className="p-6">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-emerald-400/30 bg-emerald-400/10 text-emerald-300">
                <p.icon size={20} />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-white">{p.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{p.body}</p>
            </GlassCard>
          ))}
        </div>
      </Section>

      <Section className="relative z-10 mt-24">
        <GlassCard className="relative overflow-hidden p-10 text-center">
          {/* Nano banana easter egg merged into the bottom card */}
          <img src="/nano-banana.png" alt="" className="site-merged-img site-merged-img-bottom" style={{ mixBlendMode: "screen", opacity: 0.25 }} />
          
          <Reticles className="text-emerald-400/30" />
          <Tag>// Privacy by design</Tag>
          <h3 className="relative z-10 mx-auto mt-4 max-w-2xl text-2xl font-semibold text-white sm:text-3xl">
            Powerful capability, bounded by accountability.
          </h3>
          <p className="relative z-10 mx-auto mt-3 max-w-xl text-slate-400">
            Read how SURVEILLANT keeps a human in control of every consequential decision.
          </p>
          <div className="relative z-10 mt-7">
            <Link to="/ethics" className="site-btn site-btn-primary">
              Ethics & responsible AI <ArrowRight size={16} />
            </Link>
          </div>
        </GlassCard>
      </Section>
    </div>
  );
}
