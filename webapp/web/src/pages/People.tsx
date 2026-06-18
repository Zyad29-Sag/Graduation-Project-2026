import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listPersons, type PersonListParams } from "../api/client";
import { Empty, PageHeader, Spinner } from "../components/ui";
import { PersonCard } from "../components/PersonCard";
import { PersonDrawer } from "../components/PersonDrawer";

const STATUSES = ["", "confirmed", "multi_view", "unverified", "flagged"];
const GENDERS = ["", "Male", "Female"];
const GLASSES = ["", "Glasses", "No Glasses"];
const ETHN = ["", "Asian", "Black", "Indian", "Latino_Hispanic", "Middle_Eastern", "White"];

function Select({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  placeholder: string;
}) {
  return (
    <select className="input w-auto" value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => (
        <option key={o} value={o}>
          {o === "" ? placeholder : o}
        </option>
      ))}
    </select>
  );
}

export default function People() {
  const [f, setF] = useState<PersonListParams>({});
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<string | null>(null);

  const params: PersonListParams = { ...f, q: q || undefined, limit: 120 };
  const { data, isLoading } = useQuery({
    queryKey: ["persons", params],
    queryFn: () => listPersons(params),
  });

  const upd = (k: keyof PersonListParams, v: string) =>
    setF((prev) => ({ ...prev, [k]: v || undefined }));

  return (
    <>
      <PageHeader
        title="People"
        subtitle="The identity database"
        right={<span className="chip">{data?.total ?? 0} matches</span>}
      />

      <div className="mb-5 flex flex-wrap items-center gap-2">
        <input
          className="input w-56"
          placeholder="Search id or name…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <Select value={f.status ?? ""} onChange={(v) => upd("status", v)} options={STATUSES} placeholder="Any status" />
        <Select value={f.gender ?? ""} onChange={(v) => upd("gender", v)} options={GENDERS} placeholder="Any gender" />
        <Select value={f.glasses ?? ""} onChange={(v) => upd("glasses", v)} options={GLASSES} placeholder="Glasses?" />
        <Select value={f.ethnicity ?? ""} onChange={(v) => upd("ethnicity", v)} options={ETHN} placeholder="Any ethnicity" />
      </div>

      {isLoading ? (
        <div className="grid h-64 place-items-center">
          <Spinner className="text-emerald" />
        </div>
      ) : data && data.items.length > 0 ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {data.items.map((p) => (
            <PersonCard key={p.person_id} p={p} onClick={() => setSel(p.person_id)} />
          ))}
        </div>
      ) : (
        <Empty>No people match these filters.</Empty>
      )}

      <PersonDrawer personId={sel} onClose={() => setSel(null)} />
    </>
  );
}
