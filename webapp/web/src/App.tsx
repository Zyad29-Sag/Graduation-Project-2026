import { type ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth/AuthContext";
import Layout from "./components/Layout";
import { Spinner } from "./components/ui";
import Login from "./pages/Login";
import Overview from "./pages/Overview";
import People from "./pages/People";
import SearchPage from "./pages/Search";
import LiveCams from "./pages/LiveCams";
import Alerts from "./pages/Alerts";
import Assistant from "./pages/Assistant";

function Protected({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  if (loading)
    return (
      <div className="grid h-full place-items-center">
        <Spinner className="text-emerald" />
      </div>
    );
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        element={
          <Protected>
            <Layout />
          </Protected>
        }
      >
        <Route path="/" element={<Overview />} />
        <Route path="/cameras" element={<LiveCams />} />
        <Route path="/people" element={<People />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/assistant" element={<Assistant />} />
        <Route path="/alerts" element={<Alerts />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
