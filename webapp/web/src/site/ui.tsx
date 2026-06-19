import { ReactNode, useEffect, useRef, useState } from "react";

export function Eyebrow({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`site-eyebrow ${className}`}>{children}</div>;
}

export function Tag({ children }: { children: ReactNode }) {
  return <span className="site-tag">{children}</span>;
}

export function Reticles({ className = "text-emerald-400/40" }: { className?: string }) {
  return (
    <div className={`pointer-events-none absolute inset-0 ${className}`}>
      <span className="site-reticle site-reticle-tl" />
      <span className="site-reticle site-reticle-tr" />
      <span className="site-reticle site-reticle-bl" />
      <span className="site-reticle site-reticle-br" />
    </div>
  );
}

/* ── Interactive Cursor Spotlight GlassCard ───────────────────── */
export function GlassCard({
  children,
  className = "",
  hover = false,
}: {
  children: ReactNode;
  className?: string;
  hover?: boolean;
}) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    setPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`site-glass relative overflow-hidden rounded-2xl ${hover ? "site-card-hover" : ""} ${className}`}
    >
      {/* Spotlight overlay */}
      <div
        className="pointer-events-none absolute -inset-px z-0 rounded-2xl opacity-0 transition-opacity duration-300"
        style={{
          opacity: isHovered ? 1 : 0,
          background: `radial-gradient(600px circle at ${pos.x}px ${pos.y}px, rgba(52, 211, 153, 0.08), transparent 40%)`,
        }}
      />
      {/* Inner border highlight */}
      <div
        className="pointer-events-none absolute inset-0 z-0 rounded-2xl transition-opacity duration-300"
        style={{
          opacity: isHovered ? 1 : 0,
          background: `radial-gradient(400px circle at ${pos.x}px ${pos.y}px, rgba(52, 211, 153, 0.2), transparent 40%)`,
          WebkitMask: "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
          WebkitMaskComposite: "xor",
          maskComposite: "exclude",
          padding: "1px",
        }}
      />
      <div className="relative z-10 h-full">{children}</div>
    </div>
  );
}

export function Bar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className="site-bar">
      <span style={{ width: `${pct}%` }} />
    </div>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  intro,
  center = false,
}: {
  eyebrow: string;
  title: ReactNode;
  intro?: ReactNode;
  center?: boolean;
}) {
  return (
    <div className={`${center ? "mx-auto max-w-2xl text-center" : "max-w-2xl"}`}>
      <Eyebrow>{eyebrow}</Eyebrow>
      <h2 className="mt-3 text-3xl font-semibold tracking-tight text-white sm:text-4xl">{title}</h2>
      {intro && <p className="mt-4 text-base leading-relaxed text-slate-400">{intro}</p>}
    </div>
  );
}

/* Page wrapper that gives every section consistent horizontal padding + max width */
export function Section({
  children,
  className = "",
  id,
}: {
  children: ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <section id={id} className={`mx-auto w-full max-w-6xl px-5 sm:px-8 ${className}`}>
      {children}
    </section>
  );
}
