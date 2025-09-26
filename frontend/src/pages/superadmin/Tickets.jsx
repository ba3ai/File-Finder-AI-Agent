import React, { useEffect, useState } from "react";

export default function Tickets() {
  const [status, setStatus] = useState("");
  const [items, setItems] = useState([]);

  const fetchTickets = async () => {
    const qs = new URLSearchParams();
    if (status) qs.set("status", status);
    const res = await fetch(`/api/admin/tickets?${qs.toString()}`, { credentials: "include" });
    const j = await res.json();
    setItems(j.items || []);
  };

  useEffect(()=>{ fetchTickets(); /* eslint-disable-next-line */ }, []);
  useEffect(()=>{ fetchTickets(); /* eslint-disable-next-line */ }, [status]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Tickets</h2>
        <select className="px-3 py-2 rounded-lg border" value={status} onChange={(e)=>setStatus(e.target.value)}>
          <option value="">All</option>
          <option value="open">Open</option>
          <option value="in_progress">In Progress</option>
          <option value="closed">Closed</option>
        </select>
      </div>

      <div className="grid gap-3">
        {items.length === 0 ? <div className="p-3 rounded-lg border">No tickets.</div> :
          items.map(t => (
            <div key={t.id} className="p-4 rounded-xl border bg-slate-50/60">
              <div className="flex items-center justify-between">
                <div className="font-medium">{t.title}</div>
                <div className="text-xs text-slate-600">{new Date(t.created_at).toLocaleString()}</div>
              </div>
              <div className="mt-2 flex items-center gap-2 text-sm">
                <span className={`px-2 py-1 rounded ${t.priority==="high"?"bg-rose-100 text-rose-700":t.priority==="medium"?"bg-amber-100 text-amber-700":"bg-slate-200 text-slate-700"}`}>{t.priority}</span>
                <span className={`px-2 py-1 rounded ${t.status==="open"?"bg-emerald-100 text-emerald-700":t.status==="in_progress"?"bg-blue-100 text-blue-700":"bg-slate-200 text-slate-700"}`}>{t.status}</span>
                <span className="text-slate-600">Assigned: {t.user_name || "—"}</span>
              </div>
            </div>
          ))
        }
      </div>
    </div>
  );
}
