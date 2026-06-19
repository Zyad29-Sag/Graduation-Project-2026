import { Link } from "react-router-dom";
import { useEffect, useRef } from "react";
import {
  ArrowRight,
  Brain,
  Camera,
  Eye,
  Fingerprint,
  Languages,
  ScanFace,
  Shield,
  ShieldAlert,
  Workflow,
  Zap,
} from "lucide-react";
import { Bar, Eyebrow, GlassCard, Reticles, Section, SectionHeading, Tag } from "./ui";

/* ── Scroll-reveal hook ─────────────────────────────────────────── */
function useReveal() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => { if (e.isIntersecting) e.target.classList.add("site-visible"); }),
      { threshold: 0.15 }
    );
    el.querySelectorAll(".site-reveal").forEach((c) => io.observe(c));
    return () => io.disconnect();
  }, []);
  return ref;
}

/* ── Star particles component ───────────────────────────────────── */
function Stars({ count = 30 }: { count?: number }) {
  const stars = Array.from({ length: count }, (_, i) => ({
    id: i,
    top: `${Math.random() * 100}%`,
    left: `${Math.random() * 100}%`,
    dur: `${2 + Math.random() * 4}s`,
    delay: `${Math.random() * 3}s`,
    size: Math.random() > 0.7 ? 3 : 2,
  }));
  return (
    <>
      {stars.map((s) => (
        <span
          key={s.id}
          className="site-star"
          style={{
            top: s.top, left: s.left,
            width: s.size, height: s.size,
            "--dur": s.dur, "--delay": s.delay,
          } as React.CSSProperties}
        />
      ))}
    </>
  );
}

const STATS = [
  { v: "10", l: "AI modules" },
  { v: "5", l: "synced cameras" },
  { v: "512-d", l: "Re-ID embeddings" },
  { v: "CPU", l: "no GPU required" },
];

const CAPS = [
  {
    icon: Fingerprint,
    title: "Cross-camera Re-ID",
    body: "OSNet appearance embeddings + ByteTrack keep one identity stable as a person moves between non-overlapping cameras.",
  },
  {
    icon: ScanFace,
    title: "Face & demographics",
    body: "InsightFace recognition with age, gender, ethnicity and glasses — isolated from body identity so it never corrupts tracking.",
  },
  {
    icon: ShieldAlert,
    title: "Violence detection",
    body: "A CNN-LSTM watches 16-frame clips for aggression and raises alerts with snapshots and clips in real time.",
  },
  {
    icon: Languages,
    title: "Natural-language search",
    body: 'Describe someone — "a man with glasses and a dark jacket" — and a VLM-indexed semantic search finds them.',
  },
];

const PIPELINE = [
  "Capture",
  "Detect",
  "Track",
  "Embed",
  "Match",
  "Describe",
  "Search",
];

export default function Home() {
  const rootRef = useReveal();

  return (
    <div ref={rootRef}>
      {/* ───────────────── Hero with background image ───────────────── */}
      <section className="site-hero-bg relative overflow-hidden">
        {/* Background image */}
        <img
          src="/hero-banner.jpg"
          alt=""
          aria-hidden="true"
          className="absolute inset-0 h-full w-full object-cover"
        />
        {/* Dark gradient overlays */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-[#0b1120]/70 via-[#0b1120]/55 to-[#0b1120]/95" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-[#0b1120]/50 via-transparent to-[#0b1120]/50" />

        {/* Floating cosmic orbs */}
        <div className="site-orb site-orb-1" />
        <div className="site-orb site-orb-2" />

        {/* Twinkling stars */}
        <Stars count={40} />

        {/* Rotating radar sweep — bottom-left */}
        <div className="site-hero-radar" />

        {/* Radar ring pings — emanating from radar center */}
        <div className="site-ring-ping site-ring-ping-1" />
        <div className="site-ring-ping site-ring-ping-2" />
        <div className="site-ring-ping site-ring-ping-3" />

        {/* Horizontal scanning beam — sweeps top to bottom */}
        <div className="site-h-scan" />

        {/* Vertical data stream lines — falling matrix-style */}
        <div className="site-data-stream site-data-stream-1" />
        <div className="site-data-stream site-data-stream-2" />
        <div className="site-data-stream site-data-stream-3" />
        <div className="site-data-stream site-data-stream-4" />

        {/* Moving grid overlay */}
        <div className="site-grid-overlay" />

        {/* Targeting crosshair — top right area */}
        <div className="site-crosshair" style={{ top: "18%", right: "12%" }}>
          <div className="site-crosshair-dot" />
        </div>

        {/* Identity tracking brackets — floating on hero image */}
        <div className="site-track-bracket" style={{ top: "35%", left: "20%" }}>
          <div className="site-track-bracket-inner" />
        </div>
        <div className="site-track-bracket" style={{ top: "55%", right: "18%", animationDelay: "-3s" }}>
          <div className="site-track-bracket-inner" />
        </div>

        {/* Content overlay — two-column: text left, cams right */}
        <div className="relative z-10 mx-auto flex max-w-6xl flex-col px-5 pb-16 pt-20 sm:px-8 sm:pt-28 sm:pb-20">
          <div className="grid items-center gap-10 lg:grid-cols-2">
            {/* Left — text */}
            <div>
              <Tag>● system status: active</Tag>

              <h1 className="site-hero-title mt-6 text-4xl font-bold leading-[1.08] tracking-tight text-white sm:text-5xl md:text-6xl">
                Intelligent Surveillance.
                <br />
                <span className="text-emerald-400 site-glow-text">Powered by AI.</span>
              </h1>

              <p className="site-animate-in site-delay-2 mt-6 max-w-xl text-base leading-relaxed text-slate-300 sm:text-lg">
                Next-generation threat detection and behavioral analysis engine designed for
                enterprise security infrastructures. Real-time vision and audio processing at scale.
              </p>

              <div className="site-animate-in site-delay-3 mt-8 flex flex-wrap items-center gap-3">
                <Link to="/modules" className="site-btn site-btn-primary">
                  Explore the System <ArrowRight size={16} />
                </Link>
                <Link to="/team" className="site-btn site-btn-ghost">
                  Meet the Team
                </Link>
              </div>
            </div>

            {/* Right — 2×2 camera grid */}
            <div className="site-animate-in site-delay-3 relative">
              <div className="grid grid-cols-2 gap-3">
                {[
                  { id: "CAM_0", tone: "emerald", label: "NOMINAL", v: "0.12" },
                  { id: "CAM_1", tone: "cyan", label: "TRACKING", v: "P:3601b2" },
                  { id: "CAM_2", tone: "red", label: "VIOLENCE", v: "0.87" },
                  { id: "CAM_3", tone: "emerald", label: "NOMINAL", v: "0.09" },
                ].map((c) => {
                  const tone =
                    c.tone === "red"
                      ? "text-red-400 border-red-400/40"
                      : c.tone === "cyan"
                        ? "text-cyan-300 border-cyan-300/40"
                        : "text-emerald-300 border-emerald-300/40";
                  return (
                    <div
                      key={c.id}
                      className="site-glass site-card-hover relative aspect-video overflow-hidden rounded-xl"
                    >
                      <div className={`flex items-center justify-between border-b px-2.5 py-1.5 ${tone}`}>
                        <span className="site-mono text-[10px] uppercase tracking-wider">{c.id}</span>
                        <span className="site-mono text-[10px] uppercase tracking-wider">
                          [ {c.label} {c.v} ]
                        </span>
                      </div>
                      <Reticles className={tone.split(" ")[0]} />
                      <div className="site-scan" />
                      <div className="grid h-full place-items-center">
                        <Camera className={`opacity-20 ${tone.split(" ")[0]}`} size={44} />
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="site-float absolute -bottom-4 -right-3 site-glass-strong flex items-center gap-2 rounded-lg px-3 py-2">
                <span className="h-2 w-2 rounded-full bg-emerald-400 site-pulse shadow-[0_0_8px_#34d399]" />
                <span className="site-mono text-[11px] uppercase tracking-wider text-emerald-300">
                  live · 4 nodes
                </span>
              </div>
            </div>
          </div>

          {/* Stats bar — full width below both columns */}
          <div className="site-animate-in site-delay-4 mt-12 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-white/10 sm:grid-cols-4">
            {STATS.map((s) => (
              <div key={s.l} className="site-glass px-4 py-4 text-center">
                <div className="text-xl font-bold uppercase tracking-wider text-emerald-300 site-glow-text sm:text-2xl">
                  {s.v}
                </div>
                <div className="site-mono mt-1 text-[11px] uppercase tracking-wider text-slate-400">
                  {s.l}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─────── Neon divider ─────── */}
      <div className="site-neon-line" />

      {/* ───────────────── Visual Showcase — surveillance HUD ───────────────── */}
      <Section className="pt-24">
        <div className="site-reveal grid items-center gap-10 lg:grid-cols-2">
          {/* Left — surveillance image */}
          <div className="site-img-card aspect-[3/4] max-h-[520px]">
            <img
              src="/surveillance-hud.jpg"
              alt="AI-powered CCTV with person detection overlays, object recognition, and gait analysis"
            />
            <div className="site-img-overlay" />
            <Reticles className="text-cyan-300/30" />
            <div className="absolute bottom-4 left-4 right-4">
              <div className="site-glass-strong rounded-lg px-4 py-3">
                <div className="site-mono text-[10px] uppercase tracking-wider text-cyan-300">
                  [ identity verified · subject-b-501 ]
                </div>
                <div className="mt-1 text-sm text-slate-300">
                  Multi-factor verification: facial mesh, gait analysis, object detection
                </div>
              </div>
            </div>
          </div>

          {/* Right — text */}
          <div>
            <Eyebrow>// Real-time analysis</Eyebrow>
            <h2 className="mt-4 text-3xl font-bold tracking-tight text-white sm:text-4xl">
              See everything.{" "}
              <span className="text-cyan-300 site-glow-cyan">Miss nothing.</span>
            </h2>
            <p className="mt-4 max-w-lg text-base leading-relaxed text-slate-400">
              Every frame is processed through our multi-layered AI stack — detecting people,
              recognizing faces, classifying clothing, and flagging anomalies. All in real time,
              all on CPU.
            </p>
            <div className="mt-6 space-y-3">
              {[
                { icon: Eye, label: "99.8% facial mesh match accuracy", color: "text-cyan-300" },
                { icon: Shield, label: "Gait analysis & behavioral profiling", color: "text-emerald-300" },
                { icon: Zap, label: "Sub-100ms detection pipeline", color: "text-amber-300" },
              ].map((f) => (
                <div key={f.label} className="flex items-center gap-3">
                  <div className={`flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/[0.03] ${f.color}`}>
                    <f.icon size={16} />
                  </div>
                  <span className="text-sm text-slate-300">{f.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* ───────────────── Capabilities ───────────────── */}
      <Section className="pt-28">
        <div className="site-reveal">
          <SectionHeading
            eyebrow="// Core capabilities"
            title="A complete surveillance intelligence stack"
            intro="Ten cooperating AI modules turn raw camera feeds into searchable, accountable identity intelligence."
          />
        </div>
        <div className="mt-12 grid gap-5 sm:grid-cols-2">
          {CAPS.map((c, i) => (
            <div key={c.title} className={`site-reveal`} style={{ transitionDelay: `${i * 0.1}s` }}>
              <GlassCard hover className="p-6">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-emerald-400/30 bg-emerald-400/10 text-emerald-300">
                  <c.icon size={20} />
                </div>
                <h3 className="mt-4 text-lg font-semibold text-white">{c.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">{c.body}</p>
              </GlassCard>
            </div>
          ))}
        </div>
      </Section>

      {/* ─────── Neon divider ─────── */}
      <div className="mx-auto mt-24 max-w-4xl px-5">
        <div className="site-neon-line" />
      </div>

      {/* ───────────────── Body Scan Showcase ───────────────── */}
      <Section className="pt-24">
        <div className="site-reveal grid items-center gap-10 lg:grid-cols-2">
          {/* Left — text */}
          <div>
            <Eyebrow>// Full-body intelligence</Eyebrow>
            <h2 className="mt-4 text-3xl font-bold tracking-tight text-white sm:text-4xl">
              Beyond facial recognition.{" "}
              <span className="text-emerald-400 site-glow-text">Full-body AI.</span>
            </h2>
            <p className="mt-4 max-w-lg text-base leading-relaxed text-slate-400">
              SURVEILLANT doesn't just see faces. Our 512-dimensional OSNet embeddings capture
              a person's full appearance — clothing, build, gait — creating a unique signature
              that persists across every camera in the network.
            </p>
            <div className="mt-6 grid grid-cols-2 gap-3">
              {[
                { label: "Body embeddings", value: "512-d" },
                { label: "Re-ID engine", value: "OSNet" },
                { label: "Tracker", value: "ByteTrack" },
                { label: "Index", value: "FAISS" },
              ].map((s) => (
                <div key={s.label} className="site-glass rounded-lg px-3 py-2.5">
                  <div className="text-lg font-bold text-emerald-300 site-glow-text">{s.value}</div>
                  <div className="site-mono mt-0.5 text-[10px] uppercase tracking-wider text-slate-500">{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right — body scan image */}
          <div className="relative">
            <div className="site-img-card site-glow-border aspect-video">
              <img
                src="/body-scan.jpg"
                alt="Holographic full-body skeletal scan used for person re-identification"
              />
              <div className="site-img-overlay" />
            </div>
            {/* Floating radar sweep */}
            <div className="site-radar -top-10 -right-10 opacity-40" />
          </div>
        </div>
      </Section>

      {/* ───────────────── Pipeline strip ───────────────── */}
      <Section className="pt-28">
        <div className="site-reveal">
          <GlassCard className="site-glow-border overflow-hidden p-8">
            <Eyebrow>// Unified pipeline</Eyebrow>
            <h3 className="mt-3 text-2xl font-semibold text-white">From pixels to identities</h3>
            <div className="mt-8 flex flex-wrap items-center gap-2">
              {PIPELINE.map((step, i) => (
                <div key={step} className="flex items-center gap-2">
                  <span className="site-mono rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-sm text-slate-200">
                    <span className="text-emerald-400">{String(i + 1).padStart(2, "0")}</span> {step}
                  </span>
                  {i < PIPELINE.length - 1 && <ArrowRight size={16} className="text-emerald-400/40" />}
                </div>
              ))}
            </div>
            <div className="mt-8 grid gap-6 sm:grid-cols-3">
              {[
                { k: "Re-ID accuracy", v: "Clean pose-invariant separation" },
                { k: "Search latency", v: "FAISS in-memory, sub-ms / query" },
                { k: "Describe", v: "Local VLM, no cloud" },
              ].map((m, i) => (
                <div key={m.k}>
                  <div className="site-mono text-[11px] uppercase tracking-wider text-slate-500">
                    {m.k}
                  </div>
                  <div className="mt-2">
                    <Bar value={[92, 99, 80][i]} />
                  </div>
                  <div className="mt-2 text-sm text-slate-400">{m.v}</div>
                </div>
              ))}
            </div>
          </GlassCard>
        </div>
      </Section>

      {/* ─────── Neon divider ─────── */}
      <div className="mx-auto mt-24 max-w-4xl px-5">
        <div className="site-neon-line" />
      </div>

      {/* ───────────────── Holographic Interface Showcase ───────────────── */}
      <Section className="pt-24">
        <div className="site-reveal relative overflow-hidden rounded-2xl border border-white/10">
          {/* Full-width background image */}
          <img
            src="/holo-interface.jpg"
            alt="Operator interacting with a holographic AI interface displaying person analytics"
            className="h-[500px] w-full object-cover sm:h-[550px]"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-[#04070a]/90 via-[#04070a]/60 to-transparent" />
          <div className="absolute inset-0 bg-gradient-to-t from-[#04070a]/80 via-transparent to-transparent" />

          {/* Text overlay */}
          <div className="absolute inset-0 flex items-center">
            <div className="max-w-xl px-8 sm:px-12">
              <Tag>// The future is here</Tag>
              <h2 className="mt-5 text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl">
                Command your
                <br />
                <span className="text-cyan-300 site-glow-cyan">intelligence network.</span>
              </h2>
              <p className="mt-4 max-w-md text-base leading-relaxed text-slate-300">
                A conversational AI assistant lets you search, query, and manage your entire
                surveillance network using natural language — no SQL, no dashboards, just ask.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <Link to="/login" className="site-btn site-btn-primary">
                  Open the Console <ArrowRight size={16} />
                </Link>
                <Link to="/architecture" className="site-btn site-btn-ghost">
                  <Workflow size={16} /> See Architecture
                </Link>
              </div>
            </div>
          </div>

          {/* Floating particles */}
          <Stars count={15} />
        </div>
      </Section>

      {/* ───────────────── Closing CTA ───────────────── */}
      <Section className="pt-28">
        <div className="site-reveal relative">
          <GlassCard className="relative overflow-hidden p-10 text-center sm:p-14">
            <Reticles className="text-emerald-400/30" />
            {/* Cosmic orb */}
            <div className="site-orb" style={{
              width: 200, height: 200,
              background: "rgba(16, 185, 129, 0.1)",
              top: "50%", left: "50%",
              transform: "translate(-50%, -50%)",
              filter: "blur(80px)",
              position: "absolute",
            }} />
            <Brain className="mx-auto text-emerald-400 site-glow-text" size={34} />
            <h3 className="mx-auto mt-5 max-w-2xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
              Intelligence you can see, search, and trust.
            </h3>
            <p className="mx-auto mt-4 max-w-xl text-slate-400">
              Step inside the command center or read how every module works.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-3">
              <Link to="/login" className="site-btn site-btn-primary">
                Open the console <ArrowRight size={16} />
              </Link>
              <Link to="/modules" className="site-btn site-btn-ghost">
                Browse modules
              </Link>
            </div>
          </GlassCard>
        </div>
      </Section>
    </div>
  );
}
