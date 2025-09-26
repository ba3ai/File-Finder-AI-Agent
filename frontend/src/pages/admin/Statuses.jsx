import React, { useEffect, useState } from 'react'
import { adminApi } from '../../services/adminApi'

export default function Statuses(){
  const [rows, setRows] = useState([]); const [filter, setFilter] = useState('all')
  const load = () => adminApi.users.list({ status: filter==='all'? undefined : filter, size: 50 }).then(r => setRows(r.items))
  useEffect(()=>{ load() }, [filter])
  const toggle = async (u) => { await adminApi.users.update(u.id, { status: u.status==='active'?'inactive':'active' }); load() }
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <h3 className="text-lg font-semibold">User Statuses</h3>
        <select className="input max-w-48" value={filter} onChange={e=>setFilter(e.target.value)}>
          <option value="all">All</option><option value="active">Active</option><option value="inactive">Inactive</option>
        </select>
      </div>
      <div className="card">
        <table className="table">
          <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th></th></tr></thead>
          <tbody>{rows.map(u=>(
            <tr key={u.id}>
              <td>{u.name}</td><td>{u.email}</td><td>{u.role || 'User'}</td><td>{u.status}</td>
              <td><button className="btn btn-primary" onClick={()=>toggle(u)}>Set {u.status==='active'?'Inactive':'Active'}</button></td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  )
}
