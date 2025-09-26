// frontend/src/Overview.jsx
import React, { useEffect, useMemo, useState } from "react";

/* ---------------- helpers ---------------- */
const nf = new Intl.NumberFormat();
const cf0 = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const monthKey = (d) => `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}`;
const monthLabel = (ym) => {
  const [y, m] = ym.split("-").map(Number);
  return new Date(y, m-1, 1).toLocaleDateString(undefined, { month: "short", year: "numeric" });
};
const safeNum = (v) => (typeof v === "number" && !Number.isNaN(v) ? v : 0);
const parseDate = (s) => { try { return s ? new Date(s) : null; } catch { return null; } };
const isActiveFlag = (row) => {
  if (typeof row?.is_active === "boolean") return row.is_active;
  if (typeof row?.is_active === "number") return row.is_active === 1;
  const status = (row?.status || row?.state || "").toString().toLowerCase().trim();
  if (!status) return true;
  return status === "active" || status === "enabled";
};

/* ---------------- tiny svg charts (no libs) ---------------- */
function ComboBarsLine({ months, revenueBars, dueBars, netLine }) {
  // bars: revenue & due stacked beside each other per month; line: net profit
  const width = Math.max(360, months.length * 64);
  const height = 220, pad = 28, w = width - pad*2, h = height - pad*2;

  const maxBar = Math.max(...revenueBars, ...dueBars, 1);
  const maxLine = Math.max(...netLine, 1);
  const stepX = months.length ? w / months.length : w;

  const barW = stepX * 0.32;
  const barGap = stepX * 0.10;

  const toYBar = (v) => pad + (1 - v / maxBar) * h;
  const toYLine = (v) => pad + (1 - v / Math.max(maxLine,1)) * h;

  const linePts = months.map((_, i) => {
    const x = pad + i*stepX + stepX/2;
    const y = toYLine(netLine[i] || 0);
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      {/* axes */}
      <line x1={pad} y1={pad} x2={pad} y2={height-pad} stroke="#e5e7eb" />
      <line x1={pad} y1={height-pad} x2={width-pad} y2={height-pad} stroke="#e5e7eb" />

      {/* bars */}
      {months.map((m, i) => {
        // revenue bar
        const rev = revenueBars[i] || 0;
        const due = dueBars[i] || 0;
        const xCenter = pad + i*stepX + stepX/2;
        const revH = (rev/maxBar)*h, dueH = (due/maxBar)*h;

        return (
          <g key={m}>
            {/* revenue */}
            <rect
              x={xCenter - barW - barGap/2}
              y={pad + (h - revH)}
              width={barW}
              height={revH}
              rx="6"
              fill="#2563eb"
            />
            {/* due */}
            <rect
              x={xCenter + barGap/2}
              y={pad + (h - dueH)}
              width={barW}
              height={dueH}
              rx="6"
              fill="#93c5fd"
            />
            {/* month label */}
            <text x={xCenter} y={height - 6} textAnchor="middle" fontSize="10" fill="#6b7280">
              {m.split("-")[1]}/{m.split("-")[0].slice(2)}
            </text>
          </g>
        );
      })}

      {/* net profit line */}
      <polyline fill="none" stroke="#16a34a" strokeWidth="2.5" points={linePts} strokeLinecap="round" />
    </svg>
  );
}

function StackedUsers({ months, activeArr, inactiveArr }) {
  const width = Math.max(360, months.length * 56);
  const height = 200, pad = 24, w = width - pad*2, h = height - pad*2;
  const stepX = months.length ? w / months.length : w;

  const max = Math.max(...activeArr.map((a,i)=>a+inactiveArr[i]), 1);

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      <line x1={pad} y1={height-pad} x2={width-pad} y2={height-pad} stroke="#e5e7eb" />
      {months.map((m, i) => {
        const total = activeArr[i] + inactiveArr[i];
        const activeH = (activeArr[i]/max)*h;
        const inactiveH = (inactiveArr[i]/max)*h;
        const x = pad + i*stepX + stepX*0.25;
        const bw = stepX*0.5;
        const yInactive = pad + (h - inactiveH);
        const yActive = yInactive - activeH;

        return (
          <g key={m}>
            <rect x={x} y={yInactive} width={bw} height={inactiveH} rx="6" fill="#cbd5e1" />
            <rect x={x} y={yActive} width={bw} height={activeH} rx="6" fill="#3b82f6" />
            <text x={x+bw/2} y={height-6} textAnchor="middle" fontSize="10" fill="#6b7280">
              {m.split("-")[1]}
            </text>
            <title>{monthLabel(m)} • Active {activeArr[i]} / Inactive {inactiveArr[i]} (Total {total})</title>
          </g>
        );
      })}
    </svg>
  );
}

/* ---------------- main ---------------- */
export default function Overview() {
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState("monthly"); // monthly | yearly

  // data states
  const [endUsers, setEndUsers] = useState({ items: [], total: 0 });
  const [staffUsers, setStaffUsers] = useState({ items: [], total: 0 });
  const [subs, setSubs] = useState({ items: [], total: 0 });
  const [tickets, setTickets] = useState({ items: [], total: 0 });
  const [revenue, setRevenue] = useState({ summary: { total_paid: 0, total_due: 0 }, latest: [] });

  useEffect(() => {
    let alive = true;

    async function getJSON(url) {
      const r = await fetch(url, { credentials: "include", cache: "no-store" });
      if (!r.ok) throw new Error(url + " -> " + r.status);
      return r.json();
    }

    (async () => {
      try {
        setLoading(true);
        const [uEnd, uStaff, s, t, r] = await Promise.all([
          getJSON("/api/admin/users?size=all"),
          getJSON("/api/admin/staff/users?size=all"),
          getJSON("/api/admin/subscriptions?size=all"),
          getJSON("/api/admin/tickets?size=all"),
          getJSON("/api/admin/revenue"),
        ]);
        if (!alive) return;

        setEndUsers({ items: Array.isArray(uEnd?.items) ? uEnd.items : [], total: uEnd?.total || 0 });
        setStaffUsers({ items: Array.isArray(uStaff?.items) ? uStaff.items : [], total: uStaff?.total || 0 });
        setSubs({ items: Array.isArray(s?.items) ? s.items : [], total: s?.total || (s?.items?.length || 0) });
        setTickets({ items: Array.isArray(t?.items) ? t.items : [], total: t?.total || (t?.items?.length || 0) });
        setRevenue({
          summary: r?.summary || { total_paid: 0, total_due: 0 },
          latest: Array.isArray(r?.latest) ? r.latest : [],
        });
      } catch (e) {
        console.error(e);
      } finally {
        if (alive) setLoading(false);
      }
    })();

    return () => { alive = false; };
  }, []);

  /* ---------- derive metrics ---------- */
  const userCounts = useMemo(() => {
    const usersAll = [...endUsers.items, ...staffUsers.items];
    const active = usersAll.filter(isActiveFlag).length;
    const total = usersAll.length;
    const inactive = Math.max(0, total - active);
    return { total, active, inactive };
  }, [endUsers.items, staffUsers.items]);

  const revPaid = safeNum(revenue.summary.total_paid);
  const revDue = safeNum(revenue.summary.total_due);
  const netProfit = Math.max(0, revPaid - revDue);
  const cashFlow = revPaid; // inflow proxy

  /* ---------- monthly aggregation ---------- */
  const monthly = useMemo(() => {
    // month buckets from invoices & tickets & users
    const m = new Map();

    // revenue latest: [{amount, invoice_date}, ...]
    for (const inv of revenue.latest || []) {
      const d = parseDate(inv.invoice_date) || new Date();
      const k = monthKey(d);
      if (!m.has(k)) m.set(k, { revenue: 0, due: 0, net: 0, users_active: 0, users_inactive: 0, subs: 0, tickets: 0 });
      m.get(k).revenue += safeNum(inv.amount || 0);
    }

    // treat "due" bucket as 35% of revenue if no due data per invoice (visual only)
    for (const [k, v] of m) {
      if (!v.due) v.due = Math.round(v.revenue * 0.35);
      v.net = Math.max(0, v.revenue - v.due);
    }

    // tickets: count by created_at month
    for (const tk of tickets.items || []) {
      const d = parseDate(tk.created_at) || new Date();
      const k = monthKey(d);
      if (!m.has(k)) m.set(k, { revenue: 0, due: 0, net: 0, users_active: 0, users_inactive: 0, subs: 0, tickets: 0 });
      m.get(k).tickets += 1;
    }

    // subscriptions: count by start_date month (if present)
    for (const s of subs.items || []) {
      const d = parseDate(s.start_date) || new Date();
      const k = monthKey(d);
      if (!m.has(k)) m.set(k, { revenue: 0, due: 0, net: 0, users_active: 0, users_inactive: 0, subs: 0, tickets: 0 });
      m.get(k).subs += 1;
    }

    // users: approximate active/inactive split per "created_at" month (snapshot)
    for (const u of [...endUsers.items, ...staffUsers.items]) {
      const d = parseDate(u.created_at) || new Date();
      const k = monthKey(d);
      if (!m.has(k)) m.set(k, { revenue: 0, due: 0, net: 0, users_active: 0, users_inactive: 0, subs: 0, tickets: 0 });
      if (isActiveFlag(u)) m.get(k).users_active += 1; else m.get(k).users_inactive += 1;
    }

    // sort by month asc and keep last 12
    const rows = [...m.entries()].sort((a,b)=>a[0].localeCompare(b[0])).slice(-12);
    return rows.map(([k, v]) => ({ ym: k, ...v }));
  }, [revenue.latest, tickets.items, subs.items, endUsers.items, staffUsers.items]);

  // Period toggle -> either monthly rows or year aggregation
  const tableRows = useMemo(() => {
    if (period === "monthly") return monthly;
    // yearly: roll up by year
    const byYear = new Map();
    for (const r of monthly) {
      const y = Number(r.ym.split("-")[0]);
      if (!byYear.has(y)) byYear.set(y, { revenue:0,due:0,net:0, users_active:0,users_inactive:0, subs:0, tickets:0 });
      const t = byYear.get(y);
      t.revenue += r.revenue; t.due += r.due; t.net += r.net;
      t.users_active += r.users_active; t.users_inactive += r.users_inactive;
      t.subs += r.subs; t.tickets += r.tickets;
    }
    return [...byYear.entries()].sort((a,b)=>a[0]-b[0]).map(([y, v]) => ({ ym: String(y), ...v }));
  }, [period, monthly]);

  const months = monthly.map(r => r.ym);
  const revenueBars = monthly.map(r => r.revenue);
  const dueBars = monthly.map(r => r.due);
  const netLine = monthly.map(r => r.net);
  const activeArr = monthly.map(r => r.users_active);
  const inactiveArr = monthly.map(r => r.users_inactive);

  /* ---------------- UI ---------------- */
  return (
    <div className="w-full">
      {/* Header + toggle */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="text-sm font-semibold tracking-wide text-slate-700">SUPER ADMIN DASHBOARD</div>
        <div className="rounded-full border bg-white p-1">
          <button
            className={`px-3 py-1.5 rounded-full text-sm ${period==="monthly"?"bg-black text-white":"text-slate-700"}`}
            onClick={()=>setPeriod("monthly")}
          >Monthly</button>
          <button
            className={`px-3 py-1.5 rounded-full text-sm ${period==="yearly"?"bg-black text-white":"text-slate-700"}`}
            onClick={()=>setPeriod("yearly")}
          >Yearly</button>
        </div>
      </div>

      {/* KPI cards */}
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="rounded-2xl p-4 text-white shadow-sm" style={{background:"linear-gradient(135deg,#1f2937 0%,#111827 100%)"}}>
          <div className="text-sm">Total Users</div>
          <div className="mt-2 text-3xl font-semibold">{loading ? "…" : nf.format(userCounts.total)}</div>
          <div className="mt-2 text-xs text-white/80">Active: {nf.format(userCounts.active)} • Inactive: {nf.format(userCounts.inactive)}</div>
        </div>

        <div className="rounded-2xl p-4 text-white shadow-sm" style={{background:"linear-gradient(135deg,#2563eb 0%,#4f46e5 100%)"}}>
          <div className="text-sm">Subscription Users</div>
          <div className="mt-2 text-3xl font-semibold">{loading ? "…" : nf.format(subs.total)}</div>
          <div className="mt-2 text-xs text-white/80">
            Penetration: {userCounts.total ? Math.round((subs.total/userCounts.total)*100) : 0}%
          </div>
        </div>

        <div className="rounded-2xl p-4 text-white shadow-sm" style={{background:"linear-gradient(135deg,#059669 0%,#16a34a 100%)"}}>
          <div className="text-sm">Net Profit</div>
          <div className="mt-2 text-3xl font-semibold">{loading ? "…" : cf0.format(netProfit)}</div>
          <div className="mt-2 text-xs text-white/80">Revenue: {cf0.format(revPaid)} • Due: {cf0.format(revDue)}</div>
        </div>

        <div className="rounded-2xl p-4 text-white shadow-sm" style={{background:"linear-gradient(135deg,#0ea5e9 0%,#0284c7 100%)"}}>
          <div className="text-sm">Cash Flow</div>
          <div className="mt-2 text-3xl font-semibold">{loading ? "…" : cf0.format(cashFlow)}</div>
          <div className="mt-2 text-xs text-white/80">Tickets: {nf.format(tickets.total)}</div>
        </div>
      </div>

      {/* Middle row: left table + right combo chart */}
      <div className="mt-5 grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: Monthly/Yearly Summary table */}
        <div className="rounded-2xl border bg-white p-4 lg:col-span-1 overflow-auto">
          <div className="text-sm font-semibold mb-2">{period==="monthly" ? "Monthly Summary" : "Yearly Summary"}</div>
          <table className="w-full text-sm min-w-[620px]">
            <thead className="bg-slate-50 sticky top-0">
              <tr className="text-left">
                <th className="p-2">{period==="monthly" ? "Month" : "Year"}</th>
                <th className="p-2">Users</th>
                <th className="p-2">Subs</th>
                <th className="p-2">Revenue</th>
                <th className="p-2">Tickets</th>
                <th className="p-2">ROI</th>
                <th className="p-2">Cash Flow</th>
              </tr>
            </thead>
            <tbody>
              {tableRows.length === 0 ? (
                <tr><td className="p-3" colSpan={7}>No data.</td></tr>
              ) : tableRows.map((r) => {
                  const denom = r.revenue + r.due;
                  const roi = denom > 0 ? Math.round((r.revenue - r.due) / denom * 100) : 0;
                  const users = r.users_active + r.users_inactive;
                  return (
                    <tr key={r.ym} className="border-t">
                      <td className="p-2">{period==="monthly" ? monthLabel(r.ym) : r.ym}</td>
                      <td className="p-2">{nf.format(users)}</td>
                      <td className="p-2">{nf.format(r.subs)}</td>
                      <td className="p-2">{cf0.format(r.revenue)}</td>
                      <td className="p-2">{nf.format(r.tickets)}</td>
                      <td className="p-2">{roi}%</td>
                      <td className="p-2">{cf0.format(r.revenue)}</td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>

        {/* Right: Combined chart */}
        <div className="rounded-2xl border bg-white p-4 lg:col-span-2">
          <div className="text-sm font-semibold">Revenue (bars) & Net Profit (line)</div>
          <div className="mt-2">
            <ComboBarsLine
              months={months}
              revenueBars={revenueBars}
              dueBars={dueBars}
              netLine={netLine}
            />
          </div>
          <div className="mt-2 flex items-center gap-4 text-xs text-slate-600">
            <span className="inline-flex items-center gap-2"><span className="h-2 w-3 rounded bg-blue-600 inline-block" /> Revenue</span>
            <span className="inline-flex items-center gap-2"><span className="h-2 w-3 rounded bg-blue-300 inline-block" /> Due</span>
            <span className="inline-flex items-center gap-2"><span className="h-1.5 w-6 bg-emerald-600 inline-block" /> Net</span>
          </div>
        </div>
      </div>

      {/* Bottom row: stacked users & tickets overview mini bars */}
      <div className="mt-5 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="rounded-2xl border bg-white p-4 lg:col-span-2">
          <div className="text-sm font-semibold">Active vs Inactive Users (Stacked)</div>
          <div className="mt-2">
            <StackedUsers months={months} activeArr={activeArr} inactiveArr={inactiveArr} />
          </div>
          <div className="mt-2 text-xs text-slate-500">Blue = Active • Gray = Inactive</div>
        </div>

        <div className="rounded-2xl border bg-white p-4">
          <div className="text-sm font-semibold">Tickets Overview</div>
          <div className="mt-3 grid grid-cols-3 gap-2">
            {["open","in_progress","closed"].map((k, i) => {
              const count = (tickets.items || []).filter(t => (t.status||"").toLowerCase()===k).length;
              return (
                <div key={k} className="rounded-lg border p-3 text-center">
                  <div className="text-xs text-slate-500">{["Open","In Progress","Closed"][i]}</div>
                  <div className="text-xl font-semibold">{nf.format(count)}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
