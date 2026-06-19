import { Github, Linkedin, Mail } from "lucide-react";
import { Bar, GlassCard, Reticles, Section, SectionHeading, Tag } from "./ui";

/* Photos live in webapp/web/public/Team and are served at /Team/<file>. */
const TEAM = [
  {
    name: "Zeiad Emad",
    role: "Lead Engineer · Systems & Web",
    blurb:
      "Designed the system architecture and built the core body detection, tracking and cross-camera re-identification pipeline. Developed the web platform end-to-end.",
    photo: "/Team/Zeiad Emad.jpg",
    tags: ["System Architecture", "OSNet", "ByteTrack", "FAISS", "React"],
  },
  {
    name: "Zyad Tarek",
    role: "ML / DL Engineer · Face & UI",
    blurb:
      "Machine-learning & deep-learning engineer. Built the face, gender and age models, and crafted the web user interface.",
    photo: "/Team/Zyad Tarck.jpeg",
    tags: ["PyTorch", "InsightFace", "CNN", "UI / UX"],
  },
  {
    name: "Perg",
    role: "ML / DL Engineer · Threat & Attributes",
    blurb:
      "Machine-learning & deep-learning engineer. Built the violence-detection, ethnicity and emotion-detection models, and the interactive UI demo.",
    photo: "/Team/perge.jpeg",
    tags: ["PyTorch", "CNN-LSTM", "Violence", "Emotion", "Ethnicity"],
  },
  {
    name: "Mohamed Sobhy",
    role: "Audio & Speech Engineer",
    blurb:
      "Built the audio-intelligence stack with Yousef — voice detection, language identification, and real-time understanding & translation across any language.",
    photo: "/Team/Mohamed Sobhy.jpeg",
    tags: ["Speech", "ASR", "Language ID", "Translation"],
  },
  {
    name: "Yousef Emad",
    role: "Audio & Speech Engineer",
    blurb:
      "Built the audio-intelligence stack with Mohamed — speech detection, multilingual understanding and translation to any language.",
    photo: "/Team/Yousef Emad.jpeg",
    tags: ["Speech", "VAD", "Multilingual", "Translation"],
  },
];

const MILESTONES = [
  { n: "Cross-camera Re-identification", v: 95 },
  { n: "Face & demographic pipeline", v: 90 },
  { n: "Web command center", v: 100 },
  { n: "Natural-language search", v: 85 },
  { n: "Violence detection", v: 80 },
];

const INSIGHTS = [
  { t: "Identity, not category", d: "Re-ID weights trained for individual identity beat ImageNet features that only know object categories — clean score separation across pose changes." },
  { t: "Topology beats thresholds", d: "Knowing which cameras overlap lets us rescue matches a single threshold would miss, and reject ones it would wrongly accept." },
  { t: "Describe only what you see", d: "Constraining the VLM to visible attributes removes hallucinated clothing colours and makes semantic search trustworthy." },
];

const HORIZONS = [
  { t: "Audio fusion", d: "Combine ambient audio cues with visual threat signals." },
  { t: "On-device inference", d: "Quantised models for edge deployment without a server." },
  { t: "Active learning", d: "Use operator corrections to continually refine matching." },
];

function Avatar({ name, photo }: { name: string; photo?: string }) {
  // Large centred portrait with a soft emerald halo behind it.
  const ring =
    "relative h-36 w-36 rounded-full p-[3px] bg-gradient-to-br from-emerald-400/60 via-emerald-400/10 to-cyan-400/40 shadow-[0_0_36px_-8px_rgba(52,211,153,0.55)]";
  if (photo) {
    return (
      <div className={ring}>
        <img
          src={encodeURI(photo)}
          alt={name}
          className="h-full w-full rounded-full border border-[#04070a] object-cover"
        />
      </div>
    );
  }
  const initials = name
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
  return (
    <div className={ring}>
      <div className="grid h-full w-full place-items-center rounded-full bg-gradient-to-br from-emerald-500/20 to-cyan-500/10 text-3xl font-bold text-emerald-200">
        {initials}
      </div>
    </div>
  );
}

export default function Team() {
  return (
    <div className="relative">
      {/* Background image merged into the top of the page */}
      <img src="/team-ops.png" alt="" className="site-merged-img site-merged-img-top h-[500px]" />

      <Section className="relative z-10 pt-16 sm:pt-20">
        <SectionHeading
          eyebrow="// The engineering team"
          title="Meet the team"
          intro="A graduation project built by engineers who care as much about responsibility as capability."
        />
      </Section>

      <Section className="relative z-10 mt-12">
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {TEAM.map((m, i) => (
            <GlassCard key={i} hover className="p-7">
              <div className="flex flex-col items-center text-center">
                <Avatar name={m.name} photo={m.photo} />
                <h3 className="mt-5 text-xl font-bold text-white">{m.name}</h3>
                <div className="site-mono mt-1 text-[11px] uppercase tracking-wider text-emerald-300">
                  {m.role}
                </div>
                <p className="mt-3 text-sm leading-relaxed text-slate-400">{m.blurb}</p>
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  {m.tags.map((t) => (
                    <span
                      key={t}
                      className="site-mono rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11px] text-cyan-200/80"
                    >
                      {t}
                    </span>
                  ))}
                </div>
                <div className="mt-5 flex gap-2">
                  {[Github, Linkedin, Mail].map((Icon, j) => (
                    <a
                      key={j}
                      href="#"
                      className="grid h-8 w-8 place-items-center rounded-lg border border-white/10 text-slate-400 transition-colors hover:border-emerald-400/40 hover:text-emerald-300"
                    >
                      <Icon size={14} />
                    </a>
                  ))}
                </div>
              </div>
            </GlassCard>
          ))}
        </div>
      </Section>

      {/* Technical milestones */}
      <Section className="relative z-10 mt-20">
        <div className="grid gap-8 lg:grid-cols-[1fr_1.1fr] lg:items-center">
          <div>
            <Tag>// Technical milestones</Tag>
            <h3 className="mt-4 text-2xl font-semibold text-white sm:text-3xl">
              What we shipped
            </h3>
            <p className="mt-3 max-w-md text-slate-400">
              From a single-camera prototype to a multi-camera intelligence pipeline with a full
              web command center.
            </p>
          </div>
          <GlassCard className="space-y-4 p-7">
            {MILESTONES.map((m) => (
              <div key={m.n}>
                <div className="mb-1.5 flex items-center justify-between">
                  <span className="text-sm text-slate-300">{m.n}</span>
                  <span className="site-mono text-sm text-emerald-300">{m.v}%</span>
                </div>
                <Bar value={m.v} />
              </div>
            ))}
          </GlassCard>
        </div>
      </Section>

      {/* Research insights */}
      <Section className="relative z-10 mt-20">
        <SectionHeading eyebrow="// Research insights" title="What the work taught us" />
        <div className="mt-10 grid gap-5 lg:grid-cols-3">
          {INSIGHTS.map((c, i) => (
            <GlassCard key={c.t} className="p-6">
              <div className="site-index text-4xl">{String(i + 1).padStart(2, "0")}</div>
              <h3 className="mt-3 text-base font-semibold text-white">{c.t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{c.d}</p>
            </GlassCard>
          ))}
        </div>
      </Section>

      {/* Future horizons */}
      <Section className="relative z-10 mt-20">
        <SectionHeading eyebrow="// Future horizons" title="Where it goes next" />
        <div className="mt-10 grid gap-5 sm:grid-cols-3">
          {HORIZONS.map((h) => (
            <GlassCard key={h.t} hover className="p-6">
              <h3 className="text-base font-semibold text-emerald-300">{h.t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{h.d}</p>
            </GlassCard>
          ))}
        </div>
      </Section>

      <Section className="relative z-10 mt-24">
        <GlassCard className="relative overflow-hidden p-10 text-center sm:p-14">
          <Reticles className="text-emerald-400/30" />
          <h3 className="mx-auto max-w-2xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Built by engineers. <span className="text-emerald-400 site-glow-text">Designed for people.</span>
          </h3>
        </GlassCard>
      </Section>
    </div>
  );
}
