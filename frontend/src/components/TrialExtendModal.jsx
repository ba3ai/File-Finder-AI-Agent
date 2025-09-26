import { useState } from "react";

export default function TrialExtendModal({ open, onClose, user, onSuccess }) {
  const [mode, setMode] = useState("days"); // "days" | "until"
  const [days, setDays] = useState(7);
  const [until, setUntil] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  if (!open) return null;

  const submit = async () => {
    setBusy(true); setErr("");
    try {
      const body = { user_id: user?.id };
      if (mode === "days") body.days = Number(days);
      else body.until = until;

      const r = await fetch("/api/admin/trial/extend", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.error || r.statusText);
      onSuccess?.(data);
      onClose?.();
    } catch (e) {
      setErr(e.message || "Failed to extend trial");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] grid place-items-center bg-black/40">
      <div className="w-full max-w-md rounded-2xl bg-white p-5 shadow-xl">
        <div className="text-lg font-semibold">Extend Trial</div>
        <div className="mt-1 text-sm text-slate-600">{user?.name || user?.email}</div>

        <div className="mt-4">
          <div className="flex items-center gap-2 text-sm">
            <label className="inline-flex items-center gap-2">
              <input type="radio" name="mode" value="days" checked={mode==="days"} onChange={()=>setMode("days")} />
              Add days
            </label>
            <label className="inline-flex items-center gap-2 ml-4">
              <input type="radio" name="mode" value="until" checked={mode==="until"} onChange={()=>setMode("until")} />
              Set exact date
            </label>
          </div>

          {mode === "days" ? (
            <div className="mt-3">
              <label className="text-sm">Days to add</label>
              <input
                type="number"
                min={1}
                max={60}
                className="mt-1 w-full rounded-lg border p-2"
                value={days}
                onChange={(e)=>setDays(e.target.value)}
              />
              <p className="mt-1 text-xs text-slate-500">Max 60 days from today.</p>
            </div>
          ) : (
            <div className="mt-3">
              <label className="text-sm">New trial end date</label>
              <input
                type="date"
                className="mt-1 w-full rounded-lg border p-2"
                value={until}
                onChange={(e)=>setUntil(e.target.value)}
              />
              <p className="mt-1 text-xs text-slate-500">Must be in the future (≤ 60 days ahead).</p>
            </div>
          )}
        </div>

        {err && <div className="mt-3 rounded bg-red-50 p-2 text-sm text-red-700">{err}</div>}

        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onClose} className="rounded-lg border px-4 py-2 text-sm">Cancel</button>
          <button onClick={submit} disabled={busy} className="rounded-lg bg-black px-4 py-2 text-sm text-white disabled:opacity-50">
            {busy ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
