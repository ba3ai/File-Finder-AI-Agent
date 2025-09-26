import React, { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";

const Item = ({ to, children }) => (
  <NavLink
    to={to}
    className={({ isActive }) =>
      `block px-3 py-2 rounded-lg transition ${
        isActive ? "bg-white text-black" : "bg-white/5 text-white hover:bg-white/10"
      }`
    }
  >
    {children}
  </NavLink>
);

/* Small pill-style tab for the top bar */
const TopTab = ({ to, children }) => (
  <NavLink
    to={to}
    className={({ isActive }) =>
      [
        "px-3 py-1.5 rounded-full text-sm transition border",
        isActive
          ? "bg-black text-white border-black"
          : "bg-white text-slate-700 border-slate-200 hover:border-slate-300"
      ].join(" ")
    }
  >
    {children}
  </NavLink>
);

export default function SuperAdminLayout() {
  const nav = useNavigate();
  const loc = useLocation();
  const [checking, setChecking] = useState(true);

  // Ensure only staff can access
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/api/admin/staff/me", {
          credentials: "include",
          cache: "no-store",
          headers: { "Cache-Control": "no-cache" },
        });
        if (!r.ok) throw new Error("not-staff");
        setChecking(false);
      } catch {
        nav("/staff/login", { replace: true });
      }
    })();
  }, [nav]);

  // If user hits /superadmin root, auto-redirect to /superadmin/overview
  useEffect(() => {
    const p = loc.pathname.replace(/\/+$/, "");
    if (p === "/superadmin") {
      nav("/superadmin/overview", { replace: true });
    }
  }, [loc.pathname, nav]);

  if (checking) {
    return (
      <div className="min-h-screen grid place-items-center bg-slate-50">
        <div className="text-slate-600 text-sm">Checking access…</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
          <a href="/superadmin/overview" className="text-lg font-semibold hover:opacity-80">
            Super Admin
          </a>
          <a href="/" className="text-sm text-slate-600 hover:text-slate-900">← Back to App</a>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 py-6 grid grid-cols-1 md:grid-cols-[240px,1fr] gap-6">
        {/* Sidebar */}
        <aside className="md:sticky md:top-16 h-max">
          <div className="rounded-2xl border bg-[#0b1324] p-3 text-white shadow-sm">
            <div className="text-xs uppercase tracking-wide text-white/60 px-2 pb-2">Dashboard</div>
            <nav className="space-y-1">
              <Item to="/superadmin/overview">🏠 Overview</Item>
              <Item to="/superadmin/userlist">👥 User List</Item>
              <Item to="/superadmin/add-user">➕ Add User</Item>
              <Item to="/superadmin/subscriptions">💳 Subscription History</Item>
              <Item to="/superadmin/statuses">✅ User Statuses</Item>
              <Item to="/superadmin/tickets">🎫 Tickets</Item>
              <Item to="/superadmin/revenue">📈 Revenue Info</Item>
            </nav>
          </div>
        </aside>

        {/* Main */}
        <main className="min-h-[60vh]">
          <div className="rounded-2xl border bg-white p-5 shadow-sm">
            {/* Top tabs bar (Overview tab routes to /superadmin/overview) */}
            <div className="mb-4 -mt-1 overflow-x-auto">
              <div className="flex items-center gap-2">
                <TopTab to="/superadmin/overview">Overview</TopTab>
                <TopTab to="/superadmin/userlist">User List</TopTab>
              </div>
            </div>

            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
