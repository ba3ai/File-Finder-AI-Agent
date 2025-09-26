import React, { useEffect, useState } from "react";

export default function Statuses() {
  const [status, setStatus] = useState("active");
  const [data, setData] = useState({ items: [] });

  const fetchUsers = async (s) => {
    const res = await fetch(`/api/admin/users?status=${s}`, { credentials: "include" });
    const j = await res.json();
    setData(j);
  };

  useEffect(()=>{ fetchUsers(status); }, [status]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">User Statuses</h2>
        <select className="px-3 py-2 rounded-lg border" value={status} onChange={(e)=>setStatus(e.target.value)}>
          <option value="active">Active</option>
          <option value="inactive">Inactive</option>
        </select>
      </div>

      <div className="rounded-xl border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50">
            <tr className="text-left">
              <th className="p-3">Name</th>
              <th className="p-3">Email</th>
              <th className="p-3">Status</th>
            </tr>
          </thead>
          <tbody>
            {data.items.length === 0 ? <tr><td className="p-3" colSpan={3}>No users.</td></tr> :
              data.items.map(u => (
                <tr key={u.id} className="border-t">
                  <td className="p-3">{u.name}</td>
                  <td className="p-3">{u.email}</td>
                  <td className="p-3">
                    <span className={`px-2 py-1 rounded text-xs ${u.status==="active"?"bg-green-100 text-green-700":"bg-slate-200 text-slate-700"}`}>{u.status}</span>
                  </td>
                </tr>
              ))
            }
          </tbody>
        </table>
      </div>
    </div>
  );
}
