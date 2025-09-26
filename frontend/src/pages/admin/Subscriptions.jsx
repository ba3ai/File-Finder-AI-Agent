import React, { useEffect, useState } from 'react'
import { adminApi } from '../../services/adminApi'

export default function Subscriptions(){
  const [rows, setRows] = useState([])
  useEffect(()=> { adminApi.subs.list({ size: 10 }).then(r => setRows(r.items)) }, [])
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3"><h3 className="text-lg font-semibold">Subscription History</h3></div>
      <table className="table">
        <thead><tr><th>User</th><th>Plan</th><th>Price</th><th>Status</th><th>Start</th><th>End</th></tr></thead>
        <tbody>
          {rows.map(r=>(
            <tr key={r.id}>
              <td>{r.user_name} <span className="text-slate-400">({r.user_email})</span></td>
              <td>{r.plan}</td>
              <td>${r.price}</td>
              <td><span className={`px-2 py-1 rounded-lg text-xs ${r.status==='active'?'bg-green-100 text-green-700':'bg-slate-200 dark:bg-slate-700'}`}>{r.status}</span></td>
              <td>{r.start_date ? new Date(r.start_date).toLocaleDateString() : '—'}</td>
              <td>{r.end_date ? new Date(r.end_date).toLocaleDateString() : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
