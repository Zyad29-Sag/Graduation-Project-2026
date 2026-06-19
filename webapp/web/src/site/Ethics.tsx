import { Link } from "react-router-dom";
import {
  ArrowRight,
  Check,
  Eye,
  Lock,
  Scale,
  UserCheck,
} from "lucide-react";
import { GlassCard, Reticles, Section, SectionHeading, Tag } from "./ui";

const PRINCIPLES = [
  { icon: UserCheck, title: "Human oversight", body: "AI proposes; people decide. Every merge, deletion or watchlist action requires a human with the right role." },
  { icon: Lock, title: "Privacy protection", body: "Face data lives in an isolated store, secrets come from the environment, and identity data is never mixed with chat metadata." },
  { icon: Eye, title: "Transparency", body: "Every consequential action is audit-logged with who, what and when — the system is explainable, not a black box." },
  { icon: Scale, title: "Fairness", body: "Demographic estimates are treated as soft attributes for search, never as automated judgements about a person." },
];

const PRIVACY = [
  "Face embeddings are isolated from body identity and never used to merge people.",
  "No third-party cloud — description and search models run locally on CPU.",
  "Email/alert credentials are read from environment variables, never committed.",
  "Operators can delete a person and all their data in one audited action.",
];

const FAIRNESS = [
  "Demographic outputs are advisory filters, not enforcement decisions.",
  "Thresholds are calibrated to avoid collapsing distinct people into one identity.",
  "Quality gates reject low-information crops that bias embeddings.",
  "Human review is required before any identity is permanently merged.",
];

const FRAMEWORK = [
  { n: "01", t: "Lawful basis", d: "Deploy only where monitoring is authorised and disclosed." },
  { n: "02", t: "Data minimisation", d: "Keep only what the task needs; purge on request." },
  { n: "03", t: "Accountability", d: "Audit trail for every write; role-gated actions." },
  { n: "04", t: "Review", d: "Periodic human review of matches and alerts." },
];

export default function Ethics() {
  return (
    <>
      <Section className="pt-16 sm:pt-20">
        <SectionHeading
          eyebrow="// Ethics & responsible AI"
          title="Built to be trusted"
          intro="Surveillance technology carries real responsibility. SURVEILLANT is engineered so capability never outruns accountability."
        />
      </Section>

      <Section className="mt-12">
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {PRINCIPLES.map((p) => (
            <GlassCard key={p.title} hover className="p-6">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-emerald-400/30 bg-emerald-400/10 text-emerald-300">
                <p.icon size={20} />
              </div>
              <h3 className="mt-4 text-base font-semibold text-white">{p.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{p.body}</p>
            </GlassCard>
          ))}
        </div>
      </Section>

      {/* Human in the loop banner */}
      <Section className="mt-20">
        <GlassCard className="relative overflow-hidden p-8 sm:p-10">
          <Reticles className="text-emerald-400/25" />
          <div className="grid items-center gap-8 lg:grid-cols-[1.2fr_1fr]">
            <div>
              <Tag>// Human-in-the-loop</Tag>
              <h3 className="mt-4 text-2xl font-semibold text-white sm:text-3xl">
                The machine suggests. A person decides.
              </h3>
              <p className="mt-3 max-w-xl text-slate-400">
                High-stakes actions — merging identities, adding to a watchlist, deleting records —
                are always proposed, never auto-executed. A role check and a confirmation step stand
                between the model and any irreversible change.
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2">
              {["Detect", "Propose", "Human review", "Commit"].map((s, i, arr) => (
                <div key={s} className="flex items-center gap-2">
                  <span
                    className={`site-mono rounded-lg border px-3 py-2 text-sm ${
                      s === "Human review"
                        ? "border-emerald-400/50 bg-emerald-400/10 text-emerald-300"
                        : "border-white/10 bg-white/[0.03] text-slate-300"
                    }`}
                  >
                    {s}
                  </span>
                  {i < arr.length - 1 && <ArrowRight size={14} className="text-emerald-400/40" />}
                </div>
              ))}
            </div>
          </div>
        </GlassCard>
      </Section>

      {/* Two columns */}
      <Section className="mt-20">
        <div className="grid gap-5 lg:grid-cols-2">
          {[
            { title: "Privacy & data protection", items: PRIVACY },
            { title: "Fairness & bias mitigation", items: FAIRNESS },
          ].map((col) => (
            <GlassCard key={col.title} className="p-7">
              <h3 className="text-lg font-semibold text-white">{col.title}</h3>
              <ul className="mt-4 space-y-3">
                {col.items.map((it) => (
                  <li key={it} className="flex gap-3 text-sm leading-relaxed text-slate-300">
                    <Check size={16} className="mt-0.5 shrink-0 text-emerald-400" />
                    <span>{it}</span>
                  </li>
                ))}
              </ul>
            </GlassCard>
          ))}
        </div>
      </Section>

      {/* Framework */}
      <Section className="mt-20">
        <SectionHeading eyebrow="// Responsible use framework" title="Four commitments for deployment" />
        <div className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {FRAMEWORK.map((f) => (
            <GlassCard key={f.n} className="p-6">
              <div className="site-index text-4xl">{f.n}</div>
              <h3 className="mt-3 text-base font-semibold text-white">{f.t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{f.d}</p>
            </GlassCard>
          ))}
        </div>
      </Section>

      <Section className="mt-24">
        <GlassCard className="relative overflow-hidden p-10 text-center sm:p-14">
          <Reticles className="text-emerald-400/30" />
          <h3 className="mx-auto max-w-2xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Building trust through <span className="text-emerald-400 site-glow-text">responsible AI.</span>
          </h3>
          <p className="mx-auto mt-4 max-w-xl text-slate-400">
            Meet the engineers behind the system and the research that drives it.
          </p>
          <div className="mt-8">
            <Link to="/team" className="site-btn site-btn-primary">
              Meet the team <ArrowRight size={16} />
            </Link>
          </div>
        </GlassCard>
      </Section>
    </>
  );
}
