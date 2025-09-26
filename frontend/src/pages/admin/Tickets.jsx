import React, { useEffect, useState } from 'react'
import { adminApi } from '../../services/adminApi'

export default function Tickets(){
  const [rows, setRows] = useState([]); const [status, setStatus] = useState('')
  const load = () => adminApi.tickets.list({ status: status || undefined, size: 50 }).then(r => setRows(r.items))
  useEffect(()=>{ load() }, [status])
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <h3 className="text-lg font-semibold">Tickets</h3>
        <select className="input max-w-48" value={status} onChange={e=>setStatus(e.target.value)}>
          <option value="">All</option><option value="open">Open</option><option value="in_progress">In Progress</option><option value="closed">Closed</option>
        </select>
      </div>
      <div className="card">
        <table className="table">
          <thead><tr><th>Title</th><th>Priority</th><th>Status</th><th>Assignee</th><th>Created</th></tr></thead>
          <tbody>{rows.map(t=>(
            <tr key={t.id}><td>{t.title}</td><td>{t.priority}</td><td>{t.status}</td><td>{t.user_name || '—'}</td><td>{t.created_at ? new Date(t.created_at).toLocaleString() : '—'}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  )
}
