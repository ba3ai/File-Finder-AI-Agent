// Dashboard.jsx — premium Copilot-style dashboard (drop-in)
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  FiArrowRight, FiCheck, FiExternalLink, FiLogOut, FiRefreshCw, FiSettings,
} from "react-icons/fi";
import {
  SiGoogledrive,
  SiDropbox, SiBox, SiMega,
} from "react-icons/si";
import { FiCloud } from "react-icons/fi";
import { FaMicrosoft } from "react-icons/fa";
import ConnectionModal from "../components/ConnectionModal.jsx";
import ProfileModal from "../components/ProfileModal.jsx";


/*==================================*
 *  Dashboard
 *==================================*/
export default function Dashboard() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // who am I? (used to re-fetch when user changes)
  const [me, setMe] = useState(null);

  const [connections, setConnections] = useState([
    { provider: "sharepoint", connected: false },
    { provider: "onedrive", connected: false },
    { provider: "google_drive", connected: false },
    { provider: "dropbox", connected: false },
    { provider: "box", connected: false },
    { provider: "mega", connected: false },
  ]);

  // MS/Google modal
  const [modalOpen, setModalOpen] = useState(false);
  const [modalInitial, setModalInitial] = useState("home"); // 'home' | 'sharepoint'

  // MEGA modal
  const [megaOpen, setMegaOpen] = useState(false);

  // User menu + profile modal
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const userMenuRef = useRef(null);

  useEffect(() => {
    function onClickOutside(e) {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target)) {
        setUserMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const byProvider = useMemo(() => {
    const map = {};
    for (const c of connections) map[c.provider] = c;
    return map;
  }, [connections]);

  const anyConnected = useMemo(
    () => connections.some((c) => c.connected) || false,
    [connections]
  );

  const connectedCount = useMemo(
    () => connections.filter((c) => c.connected).length,
    [connections]
  );
  const totalProviders = connections.length || 6;
  const progressPct = Math.round((connectedCount / totalProviders) * 100);

  // --- helpers ---------------------------------------------------------------

  const fetchMe = useCallback(async () => {
    const res = await fetch("/api/auth/me", {
      credentials: "include",
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" },
    });
    const data = await res.json().catch(() => ({}));
    const user = data?.authenticated ? data.user : null;
    setMe(user);
    return user;
  }, []);

  const refreshConnections = useCallback(async () => {
    setLoading(true);
    setError("");
    setSuccess("");
    try {
      const res = await fetch("/api/connections", {
        credentials: "include",
        cache: "no-store",
        headers: { "Cache-Control": "no-cache" },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Failed to load connections");
      setConnections(data.connections || []);
    } catch (e) {
      setError(e.message || "Failed to load connections");
    } finally {
      setLoading(false);
    }
  }, []);

  // first load: fetch user, then connections
  useEffect(() => {
    let mounted = true;
    (async () => {
      setLoading(true);
      const user = await fetchMe();
      if (!mounted) return;
      setConnections([]); // clear anything stale
      if (user) await refreshConnections();
      setLoading(false);
    })();
    return () => {
      mounted = false;
    };
  }, [fetchMe, refreshConnections]);

  // Handle OAuth callbacks: /dashboard?connected=ms|google|dropbox|dbx|box|all
  useEffect(() => {
    const connected = searchParams.get("connected");
    if (connected === "ms") setSuccess("Microsoft connected.");
    if (connected === "google") setSuccess("Google Drive connected.");
    if (connected === "dropbox" || connected === "dbx") setSuccess("Dropbox connected.");
    if (connected === "box") setSuccess("Box connected.");
    if (connected === "all") setSuccess("All providers connected.");
    if (connected) {
      searchParams.delete("connected");
      setSearchParams(searchParams, { replace: true });
      // refresh to reflect latest status
      refreshConnections();
    }
  }, [searchParams, setSearchParams, refreshConnections]);

  // ─── Actions ───────────────────────────────────────────────────────────────
  function connectSharePoint() { setModalInitial("sharepoint"); setModalOpen(true); }
  function connectOneDrive() { setModalInitial("sharepoint"); setModalOpen(true); }
  function connectGoogleFromModal() { setModalInitial("home"); setModalOpen(true); }

  async function connectDropbox() {
    const res = await fetch("/api/connections/dropbox/authurl", { credentials: "include" });
    const data = await res.json();
    if (data.auth_url) window.location.href = data.auth_url;
  }
  async function disconnectDropbox() {
    setSaving(true); setError(""); setSuccess("");
    try {
      const res = await fetch("/api/connections/dropbox/disconnect", { method: "DELETE", credentials: "include" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.error || "Failed to disconnect Dropbox");
      await refreshConnections();
      setSuccess("Dropbox disconnected.");
    } catch (e) { setError(e.message || "Failed to disconnect Dropbox"); }
    finally { setSaving(false); }
  }

  async function connectBox() {
    const res = await fetch("/api/connections/box/authurl", { credentials: "include" });
    const data = await res.json();
    if (data.auth_url) window.location.href = data.auth_url;
  }
  async function disconnectBox() {
    setSaving(true); setError(""); setSuccess("");
    try {
      const res = await fetch("/api/connections/box/disconnect", { method: "DELETE", credentials: "include" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.error || "Failed to disconnect Box");
      await refreshConnections();
      setSuccess("Box disconnected.");
    } catch (e) { setError(e.message || "Failed to disconnect Box"); }
    finally { setSaving(false); }
  }

  async function disconnectMicrosoft(provider) {
    setSaving(true); setError(""); setSuccess("");
    try {
      const res = await fetch(`/api/connections/microsoft/disconnect/${provider}`, {
        method: "DELETE", credentials: "include", cache: "no-store", headers: { "Cache-Control": "no-cache" },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Failed to disconnect");
      await refreshConnections();
      setSuccess(`${provider === "sharepoint" ? "SharePoint" : "OneDrive"} disconnected.`);
    } catch (e) { setError(e.message || "Failed to disconnect"); }
    finally { setSaving(false); }
  }

  async function disconnectGoogle() {
    setSaving(true); setError(""); setSuccess("");
    try {
      const res = await fetch("/api/connections/google/disconnect", {
        method: "DELETE", credentials: "include", cache: "no-store", headers: { "Cache-Control": "no-cache" },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Failed to disconnect Google Drive");
      await refreshConnections();
      setSuccess("Google Drive disconnected.");
    } catch (e) { setError(e.message || "Failed to disconnect Google Drive"); }
    finally { setSaving(false); }
  }

  function openMegaModal() { setMegaOpen(true); }
  async function disconnectMega() {
    setSaving(true); setError(""); setSuccess("");
    try {
      const res = await fetch("/api/connections/mega/disconnect", { method: "DELETE", credentials: "include" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.error || "Failed to disconnect MEGA");
      await refreshConnections();
      setSuccess("MEGA disconnected.");
    } catch (e) { setError(e.message || "Failed to disconnect MEGA"); }
    finally { setSaving(false); }
  }

  async function connectAll() {
    setSaving(true); setError(""); setSuccess("");
    try {
      const res = await fetch("/api/connections/connect_all", {
        method: "POST", credentials: "include", cache: "no-store", headers: { "Cache-Control": "no-cache" },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Failed to connect all");

      if (data.redirect) { window.location.href = data.redirect; return; }
      await refreshConnections();
      if (data.done) setSuccess("All providers connected.");
      else setSuccess(data.message || "Microsoft sources connected.");
    } catch (e) { setError(e.message || "Failed to connect all"); }
    finally { setSaving(false); }
  }

  function openAssistant() { navigate("/"); }

  async function handleLogout() {
    try {
      setSaving(true);
      const res = await fetch("/api/auth/logout", {
        method: "POST", credentials: "include", cache: "no-store", headers: { "Cache-Control": "no-cache" },
      });
      try { await res.json(); } catch { }
    } finally {
      setSaving(false); setConnections([]); setMe(null); navigate("/login", { replace: true });
    }
  }

  const userInitial = useMemo(() => {
    if (!me?.email) return "U";
    return me.email.charAt(0).toUpperCase();
  }, [me]);

  // Auto-clear banners after 4s
  useEffect(() => {
    if (!error && !success) return;
    const t = setTimeout(() => { setError(""); setSuccess(""); }, 4000);
    return () => clearTimeout(t);
  }, [error, success]);

  return (
    <div className="min-h-screen theme-surface">
      {/* Sticky action bar */}
      <div className="sticky top-0 z-40 backdrop-blur supports-[backdrop-filter]:bg-[#0b1324]/70 border-b border-white/10">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3 text-white">
            <span className="text-lg font-semibold">Storage Connections</span>
            <span className="text-[11px] hidden sm:inline">
              {connectedCount} of {totalProviders} connected
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={connectAll}
              className="rounded-full bg-white text-black px-4 py-2 text-sm font-semibold hover:opacity-90 disabled:opacity-60"
              disabled={saving}
              title="Enable SharePoint & OneDrive and start Google OAuth"
            >
              {saving ? "Please wait…" : "Connect all"}
            </button>
            <button
              onClick={openAssistant}
              disabled={!anyConnected}
              className={`rounded-full px-4 py-2 text-sm font-semibold transition ${anyConnected ? "bg-[#E0C389] text-black hover:brightness-95" : "cursor-not-allowed bg-zinc-700 text-zinc-400"
                }`}
              title={anyConnected ? "Open Assistant" : "Connect a provider to continue"}
            >
              Open Assistant
            </button>

            {/* User Menu */}
            <div className="relative" ref={userMenuRef}>
              <button
                onClick={() => setUserMenuOpen(v => !v)}
                className="ml-1 inline-flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-indigo-900 to-indigo-700 text-sm font-bold ring-1 ring-white/10"
                title={me?.email || "Account"}
              >
                {userInitial}
              </button>
              {userMenuOpen && (
                <div
                  role="menu"
                  className="absolute right-0 z-50 mt-2 w-64 rounded-2xl border theme-border theme-card p-2 shadow-2xl backdrop-blur"
                >
                  <div className="px-3 py-2">
                    <div className="text-xs theme-text opacity-70">Signed in as</div>
                    <div className="truncate text-sm theme-text" title={me?.email || ""}>
                      {me?.email || "Unknown"}
                    </div>
                  </div>

                  {/* divider that adapts to theme */}
                  <div className="my-1 border-t theme-border" />

                  <button
                    role="menuitem"
                    onClick={() => { setProfileOpen(true); setUserMenuOpen(false); }}
                    className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm theme-text hover:bg-black/5"
                  >
                    <FiSettings className="opacity-80" />
                    Profile
                  </button>

                  <button
                    role="menuitem"
                    onClick={handleLogout}
                    className="mt-1 flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm
               text-red-600 hover:bg-red-500/10"
                  >
                    <FiLogOut />
                    Logout
                  </button>
                </div>


              )}
            </div>
          </div>
        </div>
      </div>

      {/* Hero */}
      <div className="relative">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_40%_at_50%_-10%,rgba(124,58,237,.25),transparent)]" />
        <div className="mx-auto max-w-6xl px-4 pt-8">
          <div className="rounded-3xl theme-border theme-tint backdrop-blur-xl p-6 shadow-2xl">
            <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
              <div>
                <h1 className="text-2xl md:text-3xl font-bold">Connect your world</h1>
                <p className="mt-1">
                  Link your cloud drives so the assistant can find files wherever they live.
                </p>
              </div>

              {/* Progress pill */}
              <div className="flex items-center gap-4">
                <div className="relative grid place-items-center w-16 h-16">
                  <svg viewBox="0 0 36 36" className="w-16 h-16 -rotate-90">
                    <path
                      d="M18 2a16 16 0 1 1 0 32a16 16 0 1 1 0-32"
                      fill="none" stroke="rgba(255,255,255,.12)" strokeWidth="4"
                    />
                    <path
                      d="M18 2a16 16 0 1 1 0 32a16 16 0 1 1 0-32"
                      fill="none" stroke="#E0C389" strokeLinecap="round" strokeWidth="4"
                      strokeDasharray={`${(progressPct / 100) * 100} ${100 - (progressPct / 100) * 100}`}
                    />
                  </svg>
                  <div className="absolute text-sm font-semibold">{progressPct}%</div>
                </div>
                <div>
                  <div className="text-sm">Connected</div>
                  <div className="text-sm">{connectedCount} of {totalProviders} providers</div>
                </div>
              </div>
            </div>

            {/* Banners */}
            {error && (
              <div className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm">
                {error}
              </div>
            )}
            {success && (
              <div className="mt-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
                {success}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="mx-auto max-w-6xl px-4 pt-8 theme-text">
        {loading ? (
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <ProviderCard
              title="SharePoint"
              subtitle="Use Microsoft Graph (delegated) to search SharePoint"
              connected={!!byProvider.sharepoint?.connected}
              accountEmail={byProvider.sharepoint?.account_email}
              onConnect={() => { setModalInitial("sharepoint"); setModalOpen(true); }}
              onDisconnect={() => disconnectMicrosoft("sharepoint")}
              accent="from-teal-400 to-cyan-400"
              icon={<FaMicrosoft size={22} />}
              saving={saving}
            />
            <ProviderCard
              title="OneDrive"
              subtitle="Use Microsoft Graph (delegated) to search OneDrive"
              connected={!!byProvider.onedrive?.connected}
              accountEmail={byProvider.onedrive?.account_email}
              onConnect={() => { setModalInitial("sharepoint"); setModalOpen(true); }}
              onDisconnect={() => disconnectMicrosoft("onedrive")}
              accent="from-sky-400 to-blue-400"
              icon={<FiCloud size={22} />}
              saving={saving}
            />
            <ProviderCard
              title="Google Drive"
              subtitle="Connect with Google OAuth for Drive read-only"
              connected={!!byProvider.google_drive?.connected}
              accountEmail={byProvider.google_drive?.account_email}
              onConnect={connectGoogleFromModal}
              onDisconnect={disconnectGoogle}
              accent="from-emerald-400 to-green-400"
              icon={<SiGoogledrive size={22} />}
              saving={saving}
            />
            <ProviderCard
              title="Dropbox"
              subtitle="Connect with Dropbox OAuth (read-only)"
              connected={!!byProvider.dropbox?.connected}
              accountEmail={byProvider.dropbox?.account_email}
              onConnect={connectDropbox}
              onDisconnect={disconnectDropbox}
              accent="from-sky-500 to-indigo-400"
              icon={<SiDropbox size={22} />}
              saving={saving}
            />
            <ProviderCard
              title="Box"
              subtitle="Connect with Box OAuth (read-only)"
              connected={!!byProvider.box?.connected}
              accountEmail={byProvider.box?.account_email}
              onConnect={connectBox}
              onDisconnect={disconnectBox}
              accent="from-blue-500 to-indigo-500"
              icon={<SiBox size={22} />}
              saving={saving}
            />
            <ProviderCard
              title="MEGA"
              subtitle="Requires session + at least one public link"
              connected={!!byProvider.mega?.connected}
              accountEmail={byProvider.mega?.account_email}
              onConnect={openMegaModal}
              onDisconnect={disconnectMega}
              accent="from-rose-400 to-pink-500"
              icon={<SiMega size={22} />}
              saving={saving}
            />
          </div>
        )}

        <div className="mt-10 rounded-2xl border border-white/10 bg-[#0f1a33]/50 p-5 text-sm text-white">
          You can connect multiple sources. When more than one source is enabled, the assistant
          will search across all of them (you can refine in chat).
        </div>
      </div>

      {/* Microsoft/Google modal */}
      {modalOpen && (
        <ConnectionModal
          open={modalOpen}
          onClose={() => {
            setModalOpen(false);
            refreshConnections();
          }}
          initial={modalInitial}
        />
      )}

      {/* MEGA modal */}
      {megaOpen && (
        <MegaModal
          open={megaOpen}
          onClose={async (refresh = false, message = "") => {
            setMegaOpen(false);
            if (refresh) await refreshConnections();
            if (message) setSuccess(message);
          }}
        />
      )}

      {/* Profile Modal */}
      {profileOpen && (
        <ProfileModal
          me={me}
          connections={connections}
          onClose={() => setProfileOpen(false)}
        />
      )}
    </div>
  );
}

/*==================================*
 *  Provider Card (brand gradient)
 *==================================*/
function ProviderCard({
  title, subtitle, connected, accountEmail,
  onConnect, onDisconnect, saving, accent, icon,
}) {
  return (
    <div className="relative rounded-3xl p-[1.5px] bg-gradient-to-br from-white/15 to-white/5">
      <div className="h-full rounded-3xl theme-tint backdrop-blur-xl p-5 theme-border shadow-2xl transition hover:-translate-y-0.5 hover:shadow-[0_20px_60px_rgba(0,0,0,.35)]">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={`grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br ${accent} text-black`}>
              {icon}
            </div>
            <div>
              <h2 className="text-lg font-semibold">{title}</h2>
              <p className="mt-1 text-sm ">{subtitle}</p>
            </div>
          </div>
          <div
            className={`ml-3 mt-1 h-2.5 w-2.5 rounded-full ${connected
              ? "bg-emerald-400 shadow-[0_0_15px_rgba(52,211,153,0.7)]"
              : "bg-zinc-500"
              }`}
            title={connected ? "Connected" : "Not connected"}
          />
        </div>

        {accountEmail && (
          <div className="mt-3 text-xs ">
            Connected as <span>{accountEmail}</span>
          </div>
        )}

        <div className="mt-5 flex gap-3">
          {connected ? (
            <button
              onClick={onDisconnect}
              disabled={saving}
              className="inline-flex items-center gap-2 rounded-xl bg-white/10 px-3 py-2 text-sm hover:bg-white/15 disabled:opacity-50"
            >
              <FiRefreshCw className="opacity-80" /> Disconnect
            </button>
          ) : (
            <button
              onClick={onConnect}
              disabled={saving}
              className="inline-flex items-center gap-2 rounded-xl bg-white text-black px-3 py-2 text-sm font-semibold hover:opacity-90 disabled:opacity-50"
            >
              <FiArrowRight /> Connect
            </button>
          )}

          {connected && (
            <a
              href="#"
              onClick={(e) => e.preventDefault()}
              className="inline-flex items-center gap-1 rounded-xl px-3 py-2 text-xs  hover:text-white"
              title="Manage in provider console"
            >
              <FiExternalLink className="opacity-70" /> Manage
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

/*==================================*
 *  Skeleton Card (loading)
 *==================================*/
function SkeletonCard() {
  return (
    <div className="rounded-3xl border border-white/10 bg-[#0f1a33]/60 p-5">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-white/10 animate-pulse" />
          <div>
            <div className="h-4 w-24 rounded bg-white/10 animate-pulse" />
            <div className="mt-2 h-3 w-40 rounded bg-white/10 animate-pulse" />
          </div>
        </div>
        <div className="h-2.5 w-2.5 rounded-full bg-white/10" />
      </div>
      <div className="mt-5 h-8 w-28 rounded bg-white/10 animate-pulse" />
    </div>
  );
}


function StatusDot({ ok }) {
  return (
    <span
      className={`inline-block h-2.5 w-2.5 rounded-full ${ok ? "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.7)]" : "bg-zinc-600"}`}
      aria-label={ok ? "Connected" : "Not connected"} title={ok ? "Connected" : "Not connected"}
    />
  );
}

/*==================================*
 *  MEGA Modal (same logic)
 *==================================*/
function MegaModal({ open, onClose }) {
  const [session, setSession] = useState("");
  const [link, setLink] = useState("");

  const [savedSession, setSavedSession] = useState(false);
  const [savedLink, setSavedLink] = useState(false);

  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  if (!open) return null;

  useEffect(() => {
    if (open) {
      setSavedSession(false);
      setSavedLink(false);
      setMsg("");
    }
  }, [open]);

  const saveSessionOnly = async () => {
    if (!session.trim()) return;
    setBusy(true); setMsg("");
    try {
      const res = await fetch("/api/connections/mega/connect", {
        method: "POST", credentials: "include", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.error || "Failed to save session");
      setSavedSession(true); setSession(""); setMsg("✅ Session saved.");
    } catch (e) { setMsg(e.message || "Failed to save session"); }
    finally { setBusy(false); }
  };

  const addLinkOnly = async () => {
    if (!link.trim()) return;
    setBusy(true); setMsg("");
    try {
      const res = await fetch("/api/connections/mega/add_link", {
        method: "POST", credentials: "include", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ link }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.error || "Failed to add link");
      setSavedLink(true); setLink(""); setMsg("✅ Link added.");
    } catch (e) { setMsg(e.message || "Failed to add link"); }
    finally { setBusy(false); }
  };

  const connectBoth = async () => {
    setBusy(true); setMsg("");
    try {
      if (!savedSession && session.trim()) {
        const r = await fetch("/api/connections/mega/connect", {
          method: "POST", credentials: "include", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session }),
        });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j?.error || "Failed to save session");
      }
      if (!savedLink && link.trim()) {
        const r2 = await fetch("/api/connections/mega/add_link", {
          method: "POST", credentials: "include", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ link }),
        });
        const j2 = await r2.json().catch(() => ({}));
        if (!r2.ok) throw new Error(j2?.error || "Failed to add link");
      }

      const haveSession = savedSession || !!session.trim();
      const haveLink = savedLink || !!link.trim();
      if (!haveSession || !haveLink) {
        setMsg("Please add BOTH a session and at least one public link.");
        setBusy(false);
        return;
      }

      await onClose(true, "MEGA connected.");
    } catch (e) {
      setMsg(e.message || "Failed to connect MEGA"); setBusy(false);
    }
  };

  const connectEnabled = (savedSession || !!session.trim()) && (savedLink || !!link.trim());

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60" onClick={() => onClose(false)} />
      <div className="relative z-[71] w-full max-w-lg rounded-2xl border border-white/10 bg-[#0f1a33] p-6 shadow-2xl">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold">Connect MEGA</h3>
          <button onClick={() => onClose(false)} className="rounded-full p-2 hover:bg-white/5" aria-label="Close">✕</button>
        </div>

        <div className="space-y-6">
          {/* Session */}
          <div>
            <div className="mb-2 flex items-center gap-2">
              <div className="text-sm font-medium">Full account (session)</div>
              {savedSession && (
                <span className="rounded-full bg-emerald-500/20 px-2 py-[2px] text-[11px] text-emerald-300">saved</span>
              )}
            </div>
            <p className="mb-2 text-xs ">
              Generate with <code>mega-executablesession</code> (MEGAcmd) and paste the session string.
            </p>
            <textarea
              value={session}
              onChange={(e) => setSession(e.target.value)}
              rows={3}
              className="w-full rounded-xl border border-white/10 bg-[#0b1124] p-2 text-sm outline-none focus:border-white/20"
              placeholder="e.g. Hk3x... (MEGA session)"
            />
            <div className="mt-2">
              <button
                onClick={saveSessionOnly}
                disabled={busy || !session.trim()}
                className="rounded-xl bg-white px-3 py-2 text-sm font-semibold text-black transition hover:opacity-90 disabled:opacity-50"
              >
                {busy ? "Saving…" : "Save session"}
              </button>
            </div>
          </div>

          {/* Public links */}
          <div>
            <div className="mb-2 flex items-center gap-2">
              <div className="text-sm font-medium">Public links</div>
              {savedLink && (
                <span className="rounded-full bg-emerald-500/20 px-2 py-[2px] text-[11px] text-emerald-300">saved</span>
              )}
            </div>
            <input
              value={link}
              onChange={(e) => setLink(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-[#0b1124] p-2 text-sm outline-none focus:border-white/20"
              placeholder="https://mega.nz/file/...#KEY or https://mega.nz/folder/...#KEY"
            />
            <div className="mt-2 flex gap-2">
              <button
                onClick={addLinkOnly}
                disabled={busy || !link.trim()}
                className="rounded-xl bg-white px-3 py-2 text-sm font-semibold text-black transition hover:opacity-90 disabled:opacity-50"
              >
                {busy ? "Working…" : "Add link"}
              </button>
            </div>
          </div>

          {msg && (
            <div className="rounded-xl border border-white/10 bg-[#0f1a33] p-2 text-xs">
              {msg}
            </div>
          )}
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <button
            onClick={() => onClose(false)}
            className="rounded-xl bg-white/10 px-4 py-2 text-sm font-semibold hover:bg-white/15"
          >
            Close
          </button>
          <button
            onClick={connectBoth}
            disabled={busy || !connectEnabled}
            className="rounded-xl bg-white px-4 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50"
            title="Requires BOTH a session and at least one link"
          >
            {busy ? "Connecting…" : "Connect"}
          </button>
        </div>
      </div>
    </div>
  );
}
