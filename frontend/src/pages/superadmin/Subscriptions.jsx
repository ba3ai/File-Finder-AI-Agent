import React, { useEffect, useState } from "react";

export default function Subscriptions() {
  const [data, setData] = useState({ items: [], page:1, size:10, total:0 });
  const [loading, setLoading] = useState(false);

  const fetchSubs = async (page=1) => {
    setLoading(true);
    const res = await fetch(`/api/admin/subscriptions?page=${page}&size=10`, { credentials: "include" });
    const j = await res.json();
    setData(j);
    setLoading(false);
  };

  useEffect(()=>{ fetchSubs(1); }, []);

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Subscription History</h2>
      <div className="rounded-xl border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50">
            <tr className="text-left">
              <th className="p-3">User</th>
              <th className="p-3">Plan</th>
              <th className="p-3">Price</th>
              <th className="p-3">Status</th>
              <th className="p-3">Start</th>
              <th className="p-3">End</th>
            </tr>
          </thead>
          <tbody>
            {loading ? <tr><td className="p-3" colSpan={6}>Loading…</td></tr> :
              data.items.map((s)=>(
                <tr key={s.id} className="border-t">
                  <td className="p-3">{s.user_name} <span className="text-xs text-slate-500">({s.user_email})</span></td>
                  <td className="p-3">{s.plan}</td>
                  <td className="p-3">${s.price}</td>
                  <td className="p-3"><span className={`px-2 py-1 rounded text-xs ${s.status==="active"?"bg-green-100 text-green-700":"bg-amber-100 text-amber-700"}`}>{s.status}</span></td>
                  <td className="p-3">{s.start_date?.slice(0,10)}</td>
                  <td className="p-3">{s.end_date?.slice(0,10) || "-"}</td>
                </tr>
              ))
            }
          </tbody>
        </table>
      </div>
      {data.total > data.size && (
        <div className="flex justify-end gap-2">
          <button className="px-3 py-1 rounded border" onClick={()=>fetchSubs(Math.max(1, data.page-1))} disabled={data.page<=1}>Prev</button>
          <button className="px-3 py-1 rounded border" onClick={()=>fetchSubs(data.page+1)} disabled={data.page*data.size>=data.total}>Next</button>
        </div>
      )}
    </div>
  );
}
