import React, { useEffect, useMemo, useState } from "react";

/* ===================== Trial Extend Modal (self-contained) ===================== */
function TrialExtendModal({ open, onClose, user, onSuccess }) {
  const [mode, setMode] = useState("days"); // "days" | "until"
  const [days, setDays] = useState(7);
  const [until, setUntil] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    // reset state when opening
    if (open) { setMode("days"); setDays(7); setUntil(""); setErr(""); setBusy(false); }
  }, [open]);

  if (!open) return null;

  const submit = async () => {
    try {
      setBusy(true);
      setErr("");
      const body = { user_id: user?.id };
      if (mode === "days") body.days = Number(days);
      else body.until = until;

      const r = await fetch("/api/admin/trial/extend", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.error || r.statusText);
      onSuccess?.(data);
      onClose?.();
    } catch (e) {
      setErr(e.message || "Failed to extend trial");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] grid place-items-center bg-black/40">
      <div className="w-full max-w-md rounded-2xl bg-white p-5 shadow-xl">
        <div className="text-lg font-semibold">Extend Trial</div>
        <div className="mt-1 text-sm text-slate-600">{user?.name || user?.email}</div>

        <div className="mt-4">
          <div className="flex items-center gap-4 text-sm">
            <label className="inline-flex items-center gap-2">
              <input type="radio" name="mode" value="days" checked={mode === "days"} onChange={() => setMode("days")} />
              Add days
            </label>
            <label className="inline-flex items-center gap-2">
              <input type="radio" name="mode" value="until" checked={mode === "until"} onChange={() => setMode("until")} />
              Set exact date
            </label>
          </div>

          {mode === "days" ? (
            <div className="mt-3">
              <label className="text-sm">Days to add</label>
              <input
                type="number"
                min={1}
                max={60}
                className="mt-1 w-full rounded-lg border p-2"
                value={days}
                onChange={(e) => setDays(e.target.value)}
              />
              <p className="mt-1 text-xs text-slate-500">Max 60 days from today.</p>
            </div>
          ) : (
            <div className="mt-3">
              <label className="text-sm">New trial end date</label>
              <input
                type="date"
                className="mt-1 w-full rounded-lg border p-2"
                value={until}
                onChange={(e) => setUntil(e.target.value)}
              />
              <p className="mt-1 text-xs text-slate-500">Must be in the future (≤ 60 days ahead).</p>
            </div>
          )}
        </div>

        {err && <div className="mt-3 rounded bg-red-50 p-2 text-sm text-red-700">{err}</div>}

        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onClose} className="rounded-lg border px-4 py-2 text-sm">
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={busy}
            className="rounded-lg bg-black px-4 py-2 text-sm text-white disabled:opacity-50"
          >
            {busy ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ============================ Helpers & Constants ============================ */
const PAGE_SIZES = [10, 20, 50, "all"];
const SORTS = [
  { key: "newest", label: "Newest (created)" },
  { key: "last_login", label: "Last login" },
  { key: "name", label: "Name (A→Z)" },
];

function classNames(...xs) { return xs.filter(Boolean).join(" "); }
function initialsOf(name = "") {
  const s = (name || "").trim(); if (!s) return "U";
  const parts = s.split(/\s+/); return (parts[0]?.[0] || "") + (parts[1]?.[0] || "");
}
function fmtDate(s) { if (!s) return "—"; try { return new Date(s).toLocaleString(); } catch { return s; } }
function computeStatus(u){ if (u?.status) return u.status; if (typeof u?.is_active === "boolean") return u.is_active ? "active":"inactive"; return "active"; }
function displayName(u){ return u?.display_name || u?.name || (u?.email ? u.email.split("@")[0] : "User"); }
function daysUntil(s) {
  if (!s) return null;
  const now = new Date();
  const end = new Date(s);
  const diff = Math.ceil((end - now) / (1000 * 60 * 60 * 24));
  return diff;
}
const STAFF_ROLE_OPTIONS = [
  { value: "", label: "All roles" },
  { value: "admin", label: "Admin" },
  { value: "superadmin", label: "SuperAdmin" },
  { value: "client_support", label: "Client Support" },
];
const USER_ROLE_OPTIONS = [
  { value: "", label: "All roles" },
  { value: "user", label: "User" },
];

/* ================================== Main =================================== */
export default function Userlist() {
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("");         // "", "active", "inactive"
  const [role, setRole] = useState("");             // "", depends on dataset
  const [sort, setSort] = useState("newest");
  const [view, setView] = useState("grid");         // "grid" | "table"
  const [page, setPage] = useState(1);
  const [size, setSize] = useState(10);
  const [dataset, setDataset] = useState("users");  // "users" | "staff"

  const [data, setData] = useState({ items: [], total: 0, page: 1, size: 10, stats: null });
  const [loading, setLoading] = useState(false);

  // trial modal state
  const [extendOpen, setExtendOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);

  // Debounced search
  const [qDebounced, setQDebounced] = useState(q);
  useEffect(() => { const t = setTimeout(() => setQDebounced(q), 300); return () => clearTimeout(t); }, [q]);

  async function fetchPeople(nextPage = page, nextSize = size) {
    setLoading(true);
    const qs = new URLSearchParams();
    if (qDebounced) qs.set("q", qDebounced);
    if (status) qs.set("status", status);
    if (role) qs.set("role", role.toLowerCase());
    if (nextSize === "all") {
      qs.set("size", "all");
      qs.set("page", "1");
    } else {
      qs.set("page", String(nextPage || 1));
      qs.set("size", String(nextSize || 10));
    }

    const path = dataset === "staff" ? "/api/admin/staff/users" : "/api/admin/users";
    const res = await fetch(`${path}?${qs.toString()}`, { credentials: "include" });
    const j = await res.json();
    setData({
      items: Array.isArray(j.items) ? j.items : [],
      total: Number(j.total || 0),
      page: Number(j.page || nextPage || 1),
      size: j.size ?? nextSize,
      stats: j.stats || null,
    });
    setLoading(false);
  }

  // refetch on any filter / dataset change
  useEffect(() => { fetchPeople(1, size); /* eslint-disable-next-line */ }, [dataset, qDebounced, status, role, size]);

  // reset role filter when dataset changes (to avoid incompatible values)
  useEffect(() => { setRole(""); }, [dataset]);

  const sortedItems = useMemo(() => {
    const arr = [...(data.items || [])];
    switch (sort) {
      case "last_login": arr.sort((a, b) => new Date(b.last_login_at || 0) - new Date(a.last_login_at || 0)); break;
      case "name":       arr.sort((a, b) => displayName(a).localeCompare(displayName(b))); break;
      case "newest":
      default:           arr.sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0));
    }
    return arr;
  }, [data.items, sort]);

  const stats = useMemo(() => {
    if (data.stats) return data.stats;
    const totalOnPage = sortedItems.length;
    const activeOnPage = sortedItems.filter(u => computeStatus(u) === "active").length;
    return { scope: "page", total: totalOnPage, active: activeOnPage, inactive: totalOnPage - activeOnPage, newLast7Days: 0 };
  }, [data.stats, sortedItems]);

  const totalPages = size === "all" ? 1 : Math.max(1, Math.ceil((data.total || 0) / (data.size || 10)));

  const HeaderStat = ({ label, value, sub }) => (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  );

  const openExtend = (user) => { setSelectedUser(user); setExtendOpen(true); };
  const onExtendSuccess = () => fetchPeople(page, size);

  const Toolbar = (
    <div className="flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between">
      <div className="flex items-center gap-2">
        {/* Dataset toggle */}
        <div className="flex rounded-xl border bg-white p-1">
          <button
            className={classNames("px-3 py-1.5 rounded-lg text-sm", dataset === "users" ? "bg-black text-white" : "text-slate-700")}
            onClick={() => { setDataset("users"); setPage(1); }}
          >End Users</button>
          <button
            className={classNames("px-3 py-1.5 rounded-lg text-sm", dataset === "staff" ? "bg-black text-white" : "text-slate-700")}
            onClick={() => { setDataset("staff"); setPage(1); }}
          >Staff</button>
        </div>

        <div className="relative">
          <input
            value={q}
            onChange={(e) => { setPage(1); setQ(e.target.value); }}
            placeholder={`Search ${dataset === "staff" ? "staff name or email" : "name or email"}…`}
            className="pl-10 pr-3 py-2 rounded-xl border w-72 bg-white"
          />
          <span className="absolute left-3 top-2.5 text-slate-400">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M15.5 14h-.79l-.28-.27A6.5 6.5 0 1 0 9.5 16a6.47 6.47 0 0 0 4.23-1.57l.27.28v.79L20 21.5 21.5 20 15.5 14zm-6 0a4.5 4.5 0 1 1 0-9 4.5 4.5 0 0 1 0 9z"/></svg>
          </span>
        </div>

        <select className="px-3 py-2 rounded-xl border bg-white" value={status} onChange={(e) => { setPage(1); setStatus(e.target.value); }}>
          <option value="">All statuses</option>
          <option value="active">Active</option>
          <option value="inactive">Inactive</option>
        </select>

        <select className="px-3 py-2 rounded-xl border bg-white" value={role} onChange={(e) => { setPage(1); setRole(e.target.value); }}>
          {(dataset === "staff" ? STAFF_ROLE_OPTIONS : USER_ROLE_OPTIONS).map(o => (
            <option key={o.value || "all"} value={o.value}>{o.label}</option>
          ))}
        </select>

        <select className="px-3 py-2 rounded-xl border bg-white" value={sort} onChange={(e) => setSort(e.target.value)}>
          {SORTS.map(s => <option key={s.key} value={s.key}>{s.label}</option>)}
        </select>
      </div>

      <div className="flex items-center gap-2">
        <select
          className="px-3 py-2 rounded-xl border bg-white"
          value={size}
          onChange={(e) => { const v = isNaN(+e.target.value) ? e.target.value : Number(e.target.value); setSize(v); setPage(1); }}
        >
          {PAGE_SIZES.map(n => <option key={String(n)} value={n}>{n === "all" ? "All" : `${n} / page`}</option>)}
        </select>

        <div className="flex rounded-xl border bg-white p-1">
          <button className={classNames("px-3 py-1.5 rounded-lg text-sm", view === "grid" ? "bg-black text-white" : "text-slate-700")} onClick={() => setView("grid")}>Grid</button>
          <button className={classNames("px-3 py-1.5 rounded-lg text-sm", view === "table" ? "bg-black text-white" : "text-slate-700")} onClick={() => setView("table")}>Table</button>
        </div>

        <a href="/superadmin/add-user" className="px-3 py-2 rounded-xl bg-black text-white text-sm">+ Add user</a>
      </div>
    </div>
  );

  const SkeletonCard = () => (
    <div className="rounded-2xl border bg-white p-4 animate-pulse">
      <div className="h-8 w-8 rounded-full bg-slate-200" />
      <div className="mt-3 h-4 w-1/2 bg-slate-200 rounded" />
      <div className="mt-2 h-3 w-2/3 bg-slate-200 rounded" />
      <div className="mt-4 h-3 w-1/3 bg-slate-200 rounded" />
    </div>
  );

  return (
    <div className="space-y-5">
      <div className="rounded-3xl p-6 bg-gradient-to-r from-indigo-50 to-white border">
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
          <div>
            <h2 className="text-2xl font-semibold">
               {dataset === "staff" ? "Staff" : "Users"}
            </h2>
            <p className="text-sm text-slate-600 mt-1">
              {dataset === "staff" ? "All staff accounts (superadmin, admin, client support)." : "All registered end users in your workspace."}
            </p>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <HeaderStat
              label={data.size === "all" ? `${dataset === "staff" ? "Staff" : "Users"} (all)` : stats.scope === "page" ? `${dataset === "staff" ? "Staff" : "Users"} (page)` : (dataset === "staff" ? "Staff" : "Users")}
              value={(stats.total ?? data.total) || 0}
              sub={data.size === "all" ? undefined : `${data.total} total`}
            />
            <HeaderStat label="Active" value={stats.active ?? 0} />
            <HeaderStat label="Inactive" value={stats.inactive ?? 0} />
            <HeaderStat label="New (7d)" value={stats.newLast7Days ?? 0} />
          </div>
        </div>
        <div className="mt-5">{Toolbar}</div>
      </div>

      {view === "grid" ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {loading
            ? Array.from({ length: size === "all" ? 12 : size }).map((_, i) => <SkeletonCard key={i} />)
            : sortedItems.length === 0 ? (
              <div className="col-span-full rounded-2xl border bg-white p-10 text-center">
                <div className="text-lg font-medium">No records found</div>
                <div className="text-sm text-slate-600 mt-1">Try a different search or clear filters.</div>
              </div>
            ) : sortedItems.map((u) => {
              const name = displayName(u);
              const email = u?.email || "";
              const statusVal = computeStatus(u);
              const roleVal = (u?.role || (dataset === "staff" ? "admin" : "user")).toString();
              const plan = dataset === "users" ? (u?.plan || "none") : null;
              const trialEnds = dataset === "users" ? (u?.trial_ends_at || null) : null;
              const trialDaysLeft = trialEnds != null ? daysUntil(trialEnds) : null;

              return (
                <div key={`${dataset}-${u.id}`} className="rounded-2xl border bg-white p-4 hover:shadow-md transition">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-slate-900 text-white grid place-items-center text-sm font-semibold">
                      {initialsOf(name)}
                    </div>
                    <div className="min-w-0">
                      <div className="font-medium truncate">{name}</div>
                      <div className="text-sm text-slate-600 truncate">{email}</div>
                    </div>
                    <div className="ml-auto">
                      <span className={classNames("px-2 py-1 rounded text-xs", statusVal === "active" ? "bg-emerald-100 text-emerald-700" : "bg-slate-200 text-slate-700")}>
                        {statusVal}
                      </span>
                    </div>
                  </div>

                  <div className="mt-4 flex flex-wrap gap-2 text-xs">
                    <span className="px-2 py-1 rounded bg-slate-100 text-slate-700">Role: {roleVal}</span>
                    {plan && <span className="px-2 py-1 rounded bg-slate-100 text-slate-700">Plan: {plan}</span>}
                    {dataset === "users" && u.subscription_active && (
                      <span className="px-2 py-1 rounded bg-indigo-100 text-indigo-700">Subscribed</span>
                    )}
                    {dataset === "users" && trialEnds && (
                      <span className={classNames(
                        "px-2 py-1 rounded",
                        (trialDaysLeft != null && trialDaysLeft <= 7) ? "bg-yellow-100 text-yellow-700" : "bg-slate-100 text-slate-700"
                      )}>
                        Trial ends: {fmtDate(trialEnds)}
                      </span>
                    )}
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
                    <div className="rounded-lg border p-2 bg-slate-50">
                      <div className="text-slate-500">Created</div>
                      <div className="mt-0.5 font-medium">{fmtDate(u.created_at)}</div>
                    </div>
                    <div className="rounded-lg border p-2 bg-slate-50">
                      <div className="text-slate-500">Last login</div>
                      <div className="mt-0.5 font-medium">{fmtDate(u.last_login_at)}</div>
                    </div>
                  </div>

                  {dataset === "users" && (
                    <div className="mt-4 flex flex-wrap items-center gap-2">
                      <a href={`/superadmin/subscriptions?user=${encodeURIComponent(email)}`} className="px-3 py-1.5 rounded-lg border bg-white text-sm">
                        View subscription
                      </a>
                      <button
                        onClick={() => openExtend(u)}
                        className="px-3 py-1.5 rounded-lg border bg-white text-sm hover:bg-slate-50"
                        title="Extend trial end date"
                      >
                        Extend Trial
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      ) : (
        <div className="rounded-2xl border overflow-hidden bg-white">
          <table className="w-full text-sm">
            <thead className="bg-slate-50">
              <tr className="text-left">
                <th className="p-3">User</th>
                <th className="p-3">Role</th>
                <th className="p-3">Status</th>
                {dataset === "users" && <th className="p-3">Plan</th>}
                {dataset === "users" && <th className="p-3">Trial ends</th>}
                <th className="p-3">Created</th>
                <th className="p-3">Last login</th>
                {dataset === "users" && <th className="p-3">Actions</th>}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                Array.from({ length: size === "all" ? 12 : size }).map((_, i) => (
                  <tr key={i} className="animate-pulse">
                    <td className="p-3"><div className="h-4 w-48 bg-slate-200 rounded" /></td>
                    <td className="p-3"><div className="h-4 w-20 bg-slate-200 rounded" /></td>
                    <td className="p-3"><div className="h-4 w-16 bg-slate-200 rounded" /></td>
                    {dataset === "users" && <td className="p-3"><div className="h-4 w-16 bg-slate-200 rounded" /></td>}
                    {dataset === "users" && <td className="p-3"><div className="h-4 w-28 bg-slate-200 rounded" /></td>}
                    <td className="p-3"><div className="h-4 w-28 bg-slate-200 rounded" /></td>
                    <td className="p-3"><div className="h-4 w-28 bg-slate-200 rounded" /></td>
                    {dataset === "users" && <td className="p-3"><div className="h-4 w-24 bg-slate-200 rounded" /></td>}
                  </tr>
                ))
              ) : sortedItems.length === 0 ? (
                <tr><td colSpan={dataset === "users" ? 8 : 6} className="p-6 text-center">No records found.</td></tr>
              ) : sortedItems.map((u) => {
                const name = displayName(u); const email = u?.email || "";
                const statusVal = computeStatus(u); const roleVal = (u?.role || (dataset === "staff" ? "admin" : "user")).toString();
                const plan = dataset === "users" ? (u?.plan || "none") : null;
                const trialEnds = dataset === "users" ? (u?.trial_ends_at || null) : null;
                const trialDaysLeft = trialEnds != null ? daysUntil(trialEnds) : null;

                return (
                  <tr key={`${dataset}-${u.id}`} className="border-t">
                    <td className="p-3">
                      <div className="flex items-center gap-3">
                        <div className="h-8 w-8 rounded-full bg-slate-900 text-white grid place-items-center text-xs font-semibold">{initialsOf(name)}</div>
                        <div className="min-w-0">
                          <div className="font-medium truncate">{name}</div>
                          <div className="text-xs text-slate-600 truncate">{email}</div>
                        </div>
                      </div>
                    </td>
                    <td className="p-3">{roleVal}</td>
                    <td className="p-3">
                      <span className={classNames("px-2 py-1 rounded text-xs", statusVal === "active" ? "bg-emerald-100 text-emerald-700" : "bg-slate-200 text-slate-700")}>{statusVal}</span>
                    </td>
                    {dataset === "users" && <td className="p-3">{plan}</td>}
                    {dataset === "users" && (
                      <td className="p-3">
                        {trialEnds ? (
                          <span
                            className={classNames(
                              "inline-flex items-center rounded-full px-2 py-0.5 text-xs",
                              (trialDaysLeft != null && trialDaysLeft <= 7) ? "bg-yellow-100 text-yellow-700" : "bg-slate-100 text-slate-700"
                            )}
                            title={fmtDate(trialEnds)}
                          >
                            {fmtDate(trialEnds)}
                          </span>
                        ) : <span className="text-slate-400">—</span>}
                      </td>
                    )}
                    <td className="p-3">{fmtDate(u.created_at)}</td>
                    <td className="p-3">{fmtDate(u.last_login_at)}</td>
                    {dataset === "users" && (
                      <td className="p-3">
                        <div className="flex flex-wrap gap-2">
                          <a href={`/superadmin/subscriptions?user=${encodeURIComponent(email)}`} className="px-3 py-1.5 rounded-lg border bg-white text-xs">Subscription</a>
                          <button
                            onClick={() => openExtend(u)}
                            className="px-3 py-1.5 rounded-lg border bg-white text-xs hover:bg-slate-50"
                            title="Extend trial end date"
                          >
                            Extend Trial
                          </button>
                        </div>
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-600">
          Showing <span className="font-medium">{sortedItems.length}</span> of{" "}
          <span className="font-medium">{data.total}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            className="px-3 py-1.5 rounded-lg border bg-white disabled:opacity-50"
            onClick={() => { const p = Math.max(1, page - 1); setPage(p); fetchPeople(p, size); }}
            disabled={size === "all" || page <= 1 || loading}
          >Prev</button>
          <div className="text-sm min-w-[80px] text-center">
            Page {size === "all" ? 1 : page} / {size === "all" ? 1 : Math.max(1, Math.ceil((data.total || 0) / (data.size || 10)))}
          </div>
          <button
            className="px-3 py-1.5 rounded-lg border bg-white disabled:opacity-50"
            onClick={() => { const p = Math.min(Math.max(1, Math.ceil((data.total || 0) / (data.size || 10))), page + 1); setPage(p); fetchPeople(p, size); }}
            disabled={size === "all" || page >= Math.max(1, Math.ceil((data.total || 0) / (data.size || 10))) || loading}
          >Next</button>
        </div>
      </div>

      {/* Trial Extend Modal */}
      <TrialExtendModal
        open={extendOpen}
        onClose={() => setExtendOpen(false)}
        user={selectedUser}
        onSuccess={onExtendSuccess}
      />
    </div>
  );
}
