import React, { useEffect, useState } from "react";

export default function Revenue() {
  const [data, setData] = useState(null);

  useEffect(()=> {
    (async () => {
      const res = await fetch("/api/admin/revenue", { credentials: "include" });
      const j = await res.json();
      setData(j);
    })();
  }, []);

  if (!data) return <div>Loading…</div>;

  const s = data.summary || {};
  const latest = data.latest || [];

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Revenue Info</h2>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="p-4 rounded-xl border bg-white">
          <div className="text-sm text-slate-600">Total Paid</div>
          <div className="text-2xl font-semibold">${(s.total_paid ?? 0).toFixed(2)}</div>
        </div>
        <div className="p-4 rounded-xl border bg-white">
          <div className="text-sm text-slate-600">Total Due</div>
          <div className="text-2xl font-semibold">${(s.total_due ?? 0).toFixed(2)}</div>
        </div>
        <div className="p-4 rounded-xl border bg-white">
          <div className="text-sm text-slate-600">Invoices</div>
          <div className="text-2xl font-semibold">{s.invoices ?? 0}</div>
        </div>
      </div>

      <div className="rounded-xl border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50">
            <tr className="text-left">
              <th className="p-3">Invoice</th>
              <th className="p-3">Amount</th>
              <th className="p-3">Status</th>
              <th className="p-3">Date</th>
            </tr>
          </thead>
          <tbody>
            {latest.map(r => (
              <tr key={r.id} className="border-t">
                <td className="p-3">{r.invoice_no}</td>
                <td className="p-3">${r.amount}</td>
                <td className="p-3">
                  <span className={`px-2 py-1 rounded text-xs ${r.status==="paid"?"bg-green-100 text-green-700":"bg-amber-100 text-amber-700"}`}>{r.status}</span>
                </td>
                <td className="p-3">{r.invoice_date?.slice(0,10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
