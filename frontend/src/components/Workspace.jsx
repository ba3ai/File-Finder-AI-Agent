import React, { useEffect, useMemo, useState } from "react";
import InviteWizard from "../components/InviteWizard.jsx";

const cx = (...a) => a.filter(Boolean).join(" ");
const first = (s = "") => (s.trim()[0] || "U").toUpperCase();
const initials = (name = "", email = "") => {
  const n = (name || "").trim();
  if (n) {
    const parts = n.split(/\s+/).slice(0, 2);
    return parts.map((p) => p[0]).join("").toUpperCase();
  }
  return first(email);
};
const prettyEmail = (e = "") => e.toLowerCase();

export default function Workspace() {
  const [ws, setWs] = useState(null);
  const [members, setMembers] = useState([]);
  const [invites, setInvites] = useState([]);
  const [requests, setRequests] = useState([]);

  const [tab, setTab] = useState("users"); // users | invites | requests
  const [filterUsers, setFilterUsers] = useState("");
  const [filterInvites, setFilterInvites] = useState("");
  const [filterReqs, setFilterReqs] = useState("");

  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [note, setNote] = useState("");
  const [inviteOpen, setInviteOpen] = useState(false);

  const loadWorkspace = async () => {
    try {
      const r = await fetch("/api/workspaces/me", { credentials: "include" });
      if (r.status === 402) {
        setErr("Team plan required. Upgrade to access shared workspaces.");
        return;
      }
      if (!r.ok) throw new Error("Failed to load workspace.");
      const j = await r.json();
      setWs(j.workspace);
      setMembers(Array.isArray(j.members) ? j.members : []);
    } catch (e) {
      setErr(e.message || "Failed to load workspace.");
    }
  };

  const loadInvites = async () => {
    try {
      const r = await fetch("/api/workspaces/invites/pending", { credentials: "include" });
      if (!r.ok) throw new Error();
      const j = await r.json();
      setInvites(Array.isArray(j.invites) ? j.invites : []);
    } catch { setInvites([]); }
  };

  const loadRequests = async () => {
    try {
      const r = await fetch("/api/workspaces/requests/pending", { credentials: "include" });
      if (!r.ok) throw new Error();
      const j = await r.json();
      setRequests(Array.isArray(j.requests) ? j.requests : []);
    } catch { setRequests([]); }
  };

  const loadAll = async () => {
    setErr(""); setNote("");
    await loadWorkspace();
    await Promise.all([loadInvites(), loadRequests()]);
  };

  useEffect(() => { loadAll(); }, []);

  const seats = useMemo(() => {
    const used = ws?.seats_used ?? members.filter((m) => m.status === "active").length;
    const limit = ws?.seats_limit ?? 0;
    const pct = limit ? Math.min(100, Math.round((used / limit) * 100)) : 0;
    return { used, limit, pct };
  }, [ws, members]);

  // Actions
  const addExisting = async () => {
    setErr(""); setNote("");
    const v = email.trim().toLowerCase();
    if (!v) return setErr("Enter an email to add a teammate.");
    setBusy(true);
    try {
      const r = await fetch("/api/workspaces/members/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email: v }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) {
        const code = j?.error || "Failed to add member.";
        setErr(code);
        return;
      }
      setEmail("");
      setNote(`Added ${v} to “${ws?.name || "Workspace"}”.`);
      await loadWorkspace();
    } catch {
      setErr("Could not add member. Please try again.");
    } finally { setBusy(false); }
  };

  const removeMember = async (id, who) => {
    if (!window.confirm(`Remove ${who}?`)) return;
    setErr(""); setNote(""); setBusy(true);
    try {
      const r = await fetch(`/api/workspaces/members/${id}`, { method: "DELETE", credentials: "include" });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) return setErr(j?.error || "Failed to remove member.");
      setNote(`Removed ${who}.`);
      await loadWorkspace();
    } catch { setErr("Could not remove member."); }
    finally { setBusy(false); }
  };

  const cancelInvite = async (id) => {
    if (!window.confirm("Cancel this invite?")) return;
    setBusy(true); setErr(""); setNote("");
    try {
      const r = await fetch(`/api/workspaces/invites/${id}`, { method: "DELETE", credentials: "include" });
      if (!r.ok) throw new Error("Failed to cancel invite.");
      setNote("Invite cancelled.");
      await loadInvites();
    } catch (e) { setErr(e.message || "Failed to cancel invite."); }
    finally { setBusy(false); }
  };

  const resendInvite = async (id) => {
    setBusy(true); setErr(""); setNote("");
    try {
      const r = await fetch(`/api/workspaces/invites/${id}/resend`, { method: "POST", credentials: "include" });
      if (!r.ok) throw new Error("Failed to resend invite.");
      setNote("Invite resent.");
    } catch (e) { setErr(e.message || "Failed to resend invite."); }
    finally { setBusy(false); }
  };

  const acceptRequest = async (id) => {
    setBusy(true); setErr(""); setNote("");
    try {
      const r = await fetch(`/api/workspaces/requests/${id}/accept`, { method: "POST", credentials: "include" });
      if (!r.ok) throw new Error("Failed to accept request.");
      setNote("Request accepted.");
      await Promise.all([loadWorkspace(), loadRequests()]);
    } catch (e) { setErr(e.message || "Failed to accept request."); }
    finally { setBusy(false); }
  };

  const rejectRequest = async (id) => {
    setBusy(true); setErr(""); setNote("");
    try {
      const r = await fetch(`/api/workspaces/requests/${id}`, { method: "DELETE", credentials: "include" });
      if (!r.ok) throw new Error("Failed to reject request.");
      setNote("Request rejected.");
      await loadRequests();
    } catch (e) { setErr(e.message || "Failed to reject request."); }
    finally { setBusy(false); }
  };

  // Early states
  if (err && !ws) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center bg-gray-950 text-white px-6">
        <div className="max-w-lg w-full rounded-2xl border border-red-500/20 bg-gradient-to-b from-red-500/10 to-transparent p-6">
          <div className="text-red-300 font-semibold">Workspace unavailable</div>
          <div className="mt-2 text-red-200/90 text-sm">{err}</div>
          <div className="mt-4">
            <a href="/pricing" className="inline-flex items-center gap-2 rounded-xl bg-white text-black px-4 py-2 text-sm">
              Upgrade plan
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M5 12h12M13 5l7 7-7 7" /></svg>
            </a>
          </div>
        </div>
      </div>
    );
  }

  if (!ws) {
    return (
      <div className="min-h-[60vh] grid place-items-center bg-gray-950 text-white">
        <div className="animate-pulse text-white/70">Loading workspace…</div>
      </div>
    );
  }

  // Filters
  const usersFiltered = members.filter((m) => {
    const q = filterUsers.trim().toLowerCase();
    if (!q) return true;
    const n = (m?.user?.name || "").toLowerCase();
    const e = (m?.user?.email || "").toLowerCase();
    const r = (m?.role || "").toLowerCase();
    return n.includes(q) || e.includes(q) || r.includes(q);
  });

  const invitesFiltered = invites.filter((i) => {
    const q = filterInvites.trim().toLowerCase();
    if (!q) return true;
    return (i?.email || "").toLowerCase().includes(q);
  });

  const requestsFiltered = requests.filter((r) => {
    const q = filterReqs.trim().toLowerCase();
    if (!q) return true;
    return (r?.email || "").toLowerCase().includes(q);
  });

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header / hero */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,#1f3b74_0%,transparent_40%),radial-gradient(circle_at_80%_0%,#0ea5e9_0%,transparent_35%)] opacity-50" />
        <div className="relative mx-auto max-w-6xl px-6 py-12">
          <div className="flex items-start justify-between gap-6">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/80">
                <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />
                Team workspace
              </div>
              <h1 className="mt-3 text-3xl md:text-4xl font-semibold tracking-tight">
                {ws.name || "Workspace"}
              </h1>
              <p className="mt-2 text-white/70 text-sm">
                Collaborate with teammates. Share access, manage seats, and work faster together.
              </p>

              <div className="mt-5 flex gap-2">
                <button onClick={() => setInviteOpen(true)} className="rounded-xl bg-emerald-500 text-black px-4 py-2 text-sm font-medium hover:opacity-90">
                  Invite members
                </button>
                <a href="/pricing" className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white/10">
                  Get more seats
                </a>
              </div>
            </div>

            {/* Seats */}
            <div className="shrink-0 w-[260px] rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-sm text-white/80">Seats</div>
              <div className="mt-1 text-lg font-semibold">
                {seats.used} / {seats.limit}
              </div>
              <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className={cx(
                    "h-full rounded-full transition-all",
                    seats.pct < 70 ? "bg-emerald-400" : seats.pct < 100 ? "bg-yellow-300" : "bg-red-400"
                  )}
                  style={{ width: `${seats.pct}%` }}
                />
              </div>
              <div className="mt-3 text-[11px] text-white/60">
                {seats.pct >= 100 ? "All seats are used." : "Invite your teammates below."}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="mx-auto max-w-6xl px-6 pb-16">
        {(note || err) && (
          <div
            role="status"
            className={cx(
              "mt-6 rounded-xl border p-3 text-sm",
              note ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-200"
                   : "border-red-500/20 bg-red-500/10 text-red-200"
            )}
          >
            {note || err}
          </div>
        )}

        {/* Quick add existing user */}
        <div className="mt-8 rounded-2xl border border-white/10 bg-white/5 p-4">
          <div className="flex flex-col md:flex-row md:items-center md:gap-3 gap-2">
            <div className="text-sm text-white/80">Add existing user</div>
            <div className="flex-1 flex gap-2">
              <input
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="teammate@company.com"
                className="w-full rounded-xl border border-white/15 bg-[#0b1324] px-3 py-2 text-sm placeholder-white/35 outline-none focus:ring-2 focus:ring-sky-500/40"
              />
              <button
                onClick={addExisting}
                disabled={busy || seats.used >= seats.limit}
                className={cx(
                  "rounded-xl px-4 py-2 text-sm font-medium transition",
                  busy || seats.used >= seats.limit ? "bg-white/20 text-white/70 cursor-not-allowed"
                                                    : "bg-white text-black hover:opacity-90"
                )}
              >
                {busy ? "Adding…" : "Add member"}
              </button>
            </div>
            <div className="text-[11px] text-white/50">
              Instantly adds existing accounts by email. Use “Invite members” to email new teammates.
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="mt-6">
          <div className="flex items-center gap-2 border-b border-white/10">
            {[
              { key: "users", label: "Users", count: members.length },
              { key: "invites", label: "Pending invites", count: invites.length },
              { key: "requests", label: "Pending requests", count: requests.length },
            ].map(t => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={cx(
                  "relative px-3 py-2 text-sm",
                  tab === t.key ? "text-white" : "text-white/60 hover:text-white/80"
                )}
              >
                {t.label}
                <span className={cx(
                  "ml-2 inline-flex h-5 min-w-[20px] items-center justify-center rounded-full border px-1 text-[11px]",
                  tab === t.key ? "border-white/20 bg-white/10" : "border-white/10 bg-white/5 text-white/70"
                )}>
                  {t.count}
                </span>
                {tab === t.key && <span className="absolute -bottom-px left-0 right-0 h-0.5 bg-white" />}
              </button>
            ))}
          </div>

          {/* Tab bodies */}
          {tab === "users" && (
            <section className="mt-4">
              <div className="mb-3">
                <input
                  value={filterUsers}
                  onChange={(e) => setFilterUsers(e.target.value)}
                  placeholder="Filter by name, email, or role"
                  className="w-full md:w-80 rounded-xl border border-white/15 bg-[#0b1324] px-3 py-2 text-sm placeholder-white/35 outline-none focus:ring-2 focus:ring-sky-500/40"
                />
              </div>
              <div className="rounded-2xl border border-white/10 overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-white/5">
                    <tr className="text-white/70">
                      <th className="p-3 text-left">User</th>
                      <th className="p-3 text-left">Account type</th>
                      <th className="p-3 text-left">Status</th>
                      <th className="p-3 text-left w-32">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/10">
                    {usersFiltered.map((m) => {
                      const n = m?.user?.name || "";
                      const e = prettyEmail(m?.user?.email || "");
                      const isOwner = m.user_id === ws.owner_user_id;
                      return (
                        <tr key={m.id} className="hover:bg-white/5">
                          <td className="p-3">
                            <div className="flex items-center gap-3">
                              <div className="grid h-9 w-9 place-items-center rounded-full bg-white/10">
                                <span className="text-xs font-semibold text-white/80">{initials(n, e)}</span>
                              </div>
                              <div>
                                <div className="font-medium">{n || e || "User"}</div>
                                {e && <div className="text-[11px] text-white/60">{e}</div>}
                              </div>
                            </div>
                          </td>
                          <td className="p-3">{isOwner ? "Owner" : (m.role || "Member")}</td>
                          <td className="p-3">
                            <span className={cx(
                              "inline-flex items-center rounded-full px-2 py-0.5 text-[11px]",
                              m.status === "active" ? "bg-emerald-400/15 text-emerald-300" : "bg-white/10 text-white/60"
                            )}>
                              {m.status}
                            </span>
                          </td>
                          <td className="p-3">
                            <button
                              disabled={isOwner}
                              onClick={() => removeMember(m.id, n || e || "this member")}
                              className={cx(
                                "rounded-lg border px-3 py-1.5 transition text-[13px]",
                                isOwner ? "border-white/10 text-white/40 cursor-not-allowed"
                                        : "border-white/15 bg-white/5 hover:bg-white text-white hover:text-black"
                              )}
                            >
                              Remove
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                    {usersFiltered.length === 0 && (
                      <tr><td colSpan={4} className="p-6 text-center text-white/60">No users match the filter.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {tab === "invites" && (
            <section className="mt-4">
              <div className="flex items-center justify-between gap-3 mb-3">
                <input
                  value={filterInvites}
                  onChange={(e) => setFilterInvites(e.target.value)}
                  placeholder="Search for invites"
                  className="w-full md:w-80 rounded-xl border border-white/15 bg-[#0b1324] px-3 py-2 text-sm placeholder-white/35 outline-none focus:ring-2 focus:ring-sky-500/40"
                />
                <button
                  onClick={() => setInviteOpen(true)}
                  className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white hover:text-black"
                >
                  Invite member
                </button>
              </div>
              <div className="rounded-2xl border border-white/10 overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-white/5">
                    <tr className="text-white/70">
                      <th className="p-3 text-left">Email</th>
                      <th className="p-3 text-left">Date invited</th>
                      <th className="p-3 text-left">Account type</th>
                      <th className="p-3 text-left w-40">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/10">
                    {invitesFiltered.map((i) => (
                      <tr key={i.id} className="hover:bg-white/5">
                        <td className="p-3">{(i.email || "").toLowerCase()}</td>
                        <td className="p-3">{i.invited_at ? new Date(i.invited_at).toLocaleDateString() : "—"}</td>
                        <td className="p-3">{i.role || "Member"}</td>
                        <td className="p-3">
                          <div className="flex gap-2">
                            <button
                              onClick={() => resendInvite(i.id)}
                              className="rounded-lg border border-white/15 bg-white/5 hover:bg-white hover:text-black px-3 py-1.5 text-[13px]"
                            >
                              Resend
                            </button>
                            <button
                              onClick={() => cancelInvite(i.id)}
                              className="rounded-lg border border-white/15 bg-white/5 hover:bg-red-500/90 hover:border-red-500/80 px-3 py-1.5 text-[13px]"
                            >
                              Cancel
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                    {invitesFiltered.length === 0 && (
                      <tr><td colSpan={4} className="p-6 text-center text-white/60">No results</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {tab === "requests" && (
            <section className="mt-4">
              <div className="flex items-center justify-between gap-3 mb-3">
                <input
                  value={filterReqs}
                  onChange={(e) => setFilterReqs(e.target.value)}
                  placeholder="Search for requests"
                  className="w-full md:w-80 rounded-xl border border-white/15 bg-[#0b1324] px-3 py-2 text-sm placeholder-white/35 outline-none focus:ring-2 focus:ring-sky-500/40"
                />
                {requestsFiltered.length > 0 && (
                  <button
                    onClick={async () => {
                      setBusy(true); setErr(""); setNote("");
                      try {
                        await Promise.all(requestsFiltered.map((r) =>
                          fetch(`/api/workspaces/requests/${r.id}/accept`, { method: "POST", credentials: "include" })
                        ));
                        setNote("All requests accepted.");
                        await Promise.all([loadWorkspace(), loadRequests()]);
                      } catch { setErr("Failed to accept all requests."); }
                      finally { setBusy(false); }
                    }}
                    className="rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm hover:bg-white hover:text-black"
                  >
                    Accept all
                  </button>
                )}
              </div>
              <div className="rounded-2xl border border-white/10 overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-white/5">
                    <tr className="text-white/70">
                      <th className="p-3 text-left">Email</th>
                      <th className="p-3 text-left">Account type</th>
                      <th className="p-3 text-left w-48">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/10">
                    {requestsFiltered.map((r) => (
                      <tr key={r.id} className="hover:bg-white/5">
                        <td className="p-3">{(r.email || "").toLowerCase()}</td>
                        <td className="p-3">{r.role || "Member"}</td>
                        <td className="p-3">
                          <div className="flex gap-2">
                            <button
                              onClick={() => acceptRequest(r.id)}
                              className="rounded-lg border border-white/15 bg-white/5 hover:bg-white hover:text-black px-3 py-1.5 text-[13px]"
                            >
                              Accept
                            </button>
                            <button
                              onClick={() => rejectRequest(r.id)}
                              className="rounded-lg border border-white/15 bg-white/5 hover:bg-red-500/90 hover:border-red-500/80 px-3 py-1.5 text-[13px]"
                            >
                              Reject
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                    {requestsFiltered.length === 0 && (
                      <tr><td colSpan={3} className="p-6 text-center text-white/60">No results</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>

        <div className="mt-10 text-center text-xs text-white/40">
          Need more seats?{" "}
          <a href="/pricing" className="underline decoration-dotted hover:text-white/70">Contact us or upgrade</a>.
        </div>
      </div>

      {inviteOpen && (
        <InviteWizard
          onClose={async () => {
            setInviteOpen(false);
            await Promise.all([loadWorkspace(), loadInvites()]);
          }}
        />
      )}
    </div>
  );
}
