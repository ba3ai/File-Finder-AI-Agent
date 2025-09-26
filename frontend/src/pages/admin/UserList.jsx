import React, { useEffect, useState } from 'react'
import { adminApi } from '../../services/adminApi'

export default function Overview(){
  const [users, setUsers] = useState([])
  const [total, setTotal] = useState(0)

  useEffect(()=> { adminApi.users.list({ size: 5 }).then(r => { setUsers(r.items); setTotal(r.total) }) }, [])
  const active = users.filter(u => u.status==='active').length

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card"><div className="text-slate-500 text-sm">Total Users</div><div className="text-3xl font-bold mt-1">{total}</div></div>
        <div className="card"><div className="text-slate-500 text-sm">Active Users</div><div className="text-3xl font-bold mt-1">{active}</div></div>
        <div className="card"><div className="text-slate-500 text-sm">Inactive Users</div><div className="text-3xl font-bold mt-1">{total - active}</div></div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-3"><h3 className="text-lg font-semibold">Recent Users</h3></div>
        <table className="table">
          <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th></tr></thead>
          <tbody>
            {users.map(u => (
              <tr key={u.id}>
                <td>{u.name}</td><td>{u.email}</td><td>{u.role || 'User'}</td>
                <td><span className={`px-2 py-1 rounded-lg text-xs ${u.status==='active'?'bg-green-100 text-green-700':'bg-amber-100 text-amber-700'}`}>{u.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
