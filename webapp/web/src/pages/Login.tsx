import { useState, type FormEvent } from "react";
import { Navigate } from "react-router-dom";
import { Activity } from "lucide-react";
import { useAuth } from "../auth/AuthContext";
import { Spinner } from "../components/ui";

export default function Login() {
  const { login, user } = useAuth();
  const [email, setEmail] = useState("demo@surveillant.ai");
  const [password, setPassword] = useState("demo1234");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/" replace />;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await login(email, password);
    } catch {
      setErr("Incorrect email or password.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid h-full place-items-center p-6">
      <div className="w-full max-w-sm">
        <div className="mb-8 flex flex-col items-center gap-2">
          <Activity className="text-emerald-glow" size={36} />
          <div className="text-2xl font-bold tracking-[0.3em] text-emerald-glow">SURVEILLANT</div>
          <div className="text-xs uppercase tracking-widest text-emerald-200/40">
            Command Center
          </div>
        </div>
        <form onSubmit={submit} className="card space-y-3 p-6">
          <label className="block text-xs uppercase tracking-wider text-emerald-200/50">Email</label>
          <input className="input" value={email} onChange={(e) => setEmail(e.target.value)} />
          <label className="block text-xs uppercase tracking-wider text-emerald-200/50">
            Password
          </label>
          <input
            className="input"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {err && <div className="text-sm text-red-400">{err}</div>}
          <button className="btn-primary w-full" disabled={busy}>
            {busy ? <Spinner /> : "Sign in"}
          </button>
          <p className="text-center text-xs text-emerald-200/30">
            Demo: demo@surveillant.ai / demo1234
          </p>
        </form>
      </div>
    </div>
  );
}
