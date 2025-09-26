import React, { useState } from 'react'
import { adminApi } from '../../services/adminApi'

export default function AddUser(){
  const [form, setForm] = useState({ name:'', email:'', role:'User', status:'active' })
  const [msg, setMsg] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    setMsg('')
    try { await adminApi.users.create(form); setMsg('✅ User created'); setForm({ name:'', email:'', role:'User', status:'active' }) }
    catch(e){ setMsg('❌ '+e.message) }
  }

  return (
    <div className="card max-w-xl">
      <h3 className="text-lg font-semibold mb-4">Manual User Entry</h3>
      <form onSubmit={submit} className="space-y-3">
        <div><div className="label">Name</div><input className="input" value={form.name} onChange={e=>setForm({...form, name:e.target.value})} /></div>
        <div><div className="label">Email</div><input className="input" value={form.email} onChange={e=>setForm({...form, email:e.target.value})} /></div>
        <div className="grid grid-cols-2 gap-3">
          <div><div className="label">Role</div>
            <select className="input" value={form.role} onChange={e=>setForm({...form, role:e.target.value})}>
              <option>User</option><option>Admin</option><option>Owner</option><option>Operator</option><option>Analyst</option>
            </select>
          </div>
          <div><div className="label">Status</div>
            <select className="input" value={form.status} onChange={e=>setForm({...form, status:e.target.value})}>
              <option value="active">Active</option><option value="inactive">Inactive</option>
            </select>
          </div>
        </div>
        <div className="flex items-center gap-2 pt-2"><button className="btn btn-primary" type="submit">Save</button>{msg && <span className="text-sm">{msg}</span>}</div>
      </form>
    </div>
  )
}
