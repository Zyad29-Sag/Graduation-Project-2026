import { useEffect, useRef, useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { Activity, ArrowRight, Github } from "lucide-react";
import { useAuth } from "../auth/AuthContext";
import "./site.css";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/modules", label: "Modules" },
  { to: "/architecture", label: "Architecture" },
  { to: "/ethics", label: "Ethics" },
  { to: "/team", label: "Team" },
];

function Brand() {
  return (
    <Link to="/" className="flex items-center gap-2">
      <Activity className="text-emerald-400 site-glow-text" size={22} />
      <span className="text-lg font-bold tracking-[0.16em] text-emerald-300 site-glow-text">
        SURVEILLANT
      </span>
    </Link>
  );
}

function BootSequence({ onComplete }: { onComplete: () => void }) {
  const [lines, setLines] = useState<string[]>([]);

  useEffect(() => {
    const sequence = [
      "INITIALIZING NEURAL NET...",
      "LOADING OSNET EMBEDDINGS...",
      "ESTABLISHING SECURE UPLINK...",
      "[ OK ] SYSTEM ONLINE"
    ];
    let step = 0;
    
    const interval = setInterval(() => {
      if (step < sequence.length) {
        setLines(prev => [...prev, sequence[step]]);
        step++;
      } else {
        clearInterval(interval);
        setTimeout(onComplete, 400);
      }
    }, 250);

    return () => clearInterval(interval);
  }, [onComplete]);

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-[#04070a]">
      <div className="site-mono w-full max-w-lg px-8 text-sm text-cyan-400">
        <div className="mb-4 flex items-center gap-2">
          <Activity className="site-pulse text-emerald-400" size={20} />
          <span className="text-emerald-300">SURVEILLANT_SYS_v2.0</span>
        </div>
        {lines.map((l, i) => (
          <div key={i} className="mb-1 opacity-80">{l}</div>
        ))}
        {lines.length < 4 && <div className="site-cursor mt-2" />}
      </div>
    </div>
  );
}

function NeonCursorTrail() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = window.innerWidth;
    let height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    const handleResize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };
    window.addEventListener("resize", handleResize);

    let mouse = { x: width / 2, y: height / 2 };
    let points = Array.from({ length: 40 }, () => ({ x: mouse.x, y: mouse.y }));

    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };
    window.addEventListener("mousemove", handleMouseMove);

    let animationFrameId: number;

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Spring physics for smooth trailing
      points[0].x += (mouse.x - points[0].x) * 0.4;
      points[0].y += (mouse.y - points[0].y) * 0.4;

      for (let i = 1; i < points.length; i++) {
        points[i].x += (points[i - 1].x - points[i].x) * 0.4;
        points[i].y += (points[i - 1].y - points[i].y) * 0.4;
      }

      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowBlur = 6; // Reduced from 12 for subtlety
      ctx.shadowColor = "#34d399"; // emerald-400

      // Draw segments for tapering and fading
      for (let i = 1; i < points.length; i++) {
        ctx.beginPath();
        ctx.moveTo(points[i - 1].x, points[i - 1].y);
        ctx.lineTo(points[i].x, points[i].y);
        
        const distanceRatio = 1 - i / points.length;
        const opacity = distanceRatio * 0.5; // Max opacity is 0.5 instead of 1.0
        
        // Cyan at the head, blending to Emerald at the tail
        ctx.strokeStyle = `rgba(52, 211, 153, ${opacity})`;
        ctx.lineWidth = 3.5 * Math.pow(distanceRatio, 1.5); // Reduced width
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none fixed inset-0 z-50"
      style={{ mixBlendMode: "screen" }}
    />
  );
}

export default function SiteLayout() {
  const { user } = useAuth();
  const [booting, setBooting] = useState(true);

  return (
    <>
      {booting && <BootSequence onComplete={() => setBooting(false)} />}
      <div className={`site-shell relative z-10 flex min-h-full flex-col ${booting ? "opacity-0" : "opacity-100 transition-opacity duration-1000"}`}>
        <NeonCursorTrail />
        {/* Header */}
        <header className="sticky top-0 z-50 site-glass border-b border-white/10">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 py-3.5 sm:px-8">
          <Brand />
          <nav className="hidden items-center gap-1 md:flex">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? "text-emerald-300"
                      : "text-slate-300 hover:text-white"
                  }`
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <Link to={user ? "/app" : "/login"} className="site-btn site-btn-primary !px-4 !py-2 text-sm">
            {user ? "Open console" : "Sign in"}
            <ArrowRight size={16} />
          </Link>
        </div>
      </header>

      {/* Page body */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="mt-24 border-t border-white/10 site-glass">
        <div className="mx-auto w-full max-w-6xl px-5 py-10 sm:px-8">
          <div className="flex flex-col items-start justify-between gap-6 md:flex-row md:items-center">
            <div>
              <Brand />
              <p className="mt-3 max-w-md text-sm leading-relaxed text-slate-400">
                Multi-camera AI person re-identification & tracking — a CPU-only surveillance
                intelligence pipeline. Graduation project.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <nav className="flex flex-wrap gap-x-4 gap-y-1">
                {NAV.map((n) => (
                  <Link key={n.to} to={n.to} className="text-sm text-slate-400 hover:text-emerald-300">
                    {n.label}
                  </Link>
                ))}
              </nav>
              <a
                href="#"
                className="grid h-9 w-9 place-items-center rounded-lg border border-white/10 text-slate-400 hover:text-white"
                aria-label="Repository"
              >
                <Github size={16} />
              </a>
            </div>
          </div>
          <div className="site-divider my-7" />
          <div className="flex flex-col items-center justify-between gap-2 text-center md:flex-row md:text-left">
            <p className="site-mono text-xs text-slate-500">© 2026 SURVEILLANT — All rights reserved.</p>
            <p className="site-mono text-xs text-emerald-400/60">
              [ Built by Engineers · Designed for People ]
            </p>
          </div>
        </div>
      </footer>
    </div>
    </>
  );
}
