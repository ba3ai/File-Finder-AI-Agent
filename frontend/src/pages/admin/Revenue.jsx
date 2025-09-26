import React, { useEffect, useState } from 'react'
import { adminApi } from '../../services/adminApi'

export default function Revenue(){
  const [sum, setSum] = useState({ total_paid: 0, total_due: 0, invoices: 0 })
  const [latest, setLatest] = useState([])
  useEffect(()=> { adminApi.revenue.summary().then(r => { setSum(r.summary); setLatest(r.latest) }) }, [])
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card"><div className="text-slate-500 text-sm">Total Paid</div><div className="text-3xl font-bold mt-1">${(sum.total_paid||0).toFixed(2)}</div></div>
        <div className="card"><div className="text-slate-500 text-sm">Total Due</div><div className="text-3xl font-bold mt-1">${(sum.total_due||0).toFixed(2)}</div></div>
        <div className="card"><div className="text-slate-500 text-sm">Invoices</div><div className="text-3xl font-bold mt-1">{sum.invoices}</div></div>
      </div>
      <div className="card">
        <h3 className="text-lg font-semibold mb-3">Latest Invoices</h3>
        <table className="table">
          <thead><tr><th>Invoice</th><th>Amount</th><th>Status</th><th>Date</th></tr></thead>
          <tbody>{latest.map(r=>(
            <tr key={r.id}><td>{r.invoice_no}</td><td>${r.amount}</td><td>{r.status}</td><td>{r.invoice_date ? new Date(r.invoice_date).toLocaleDateString() : '—'}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  )
}
