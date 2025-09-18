// src/components/ProfileModal.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  FiX, FiUser, FiShield, FiSun, FiMoon, FiMonitor, FiKey, FiEdit2, FiCopy,
  FiEye, FiEyeOff, FiDownload, FiTrash2, FiSmartphone, FiMail, FiCalendar
} from "react-icons/fi";
import { FaMicrosoft, FaSlack } from "react-icons/fa";
import { FiCloud } from "react-icons/fi";
import { SiGoogledrive, SiDropbox, SiBox, SiMega } from "react-icons/si";
import { PiMicrosoftTeamsLogoFill } from "react-icons/pi";

/* ────────────────────────────────────────────────────────────────────────── *
 *  Profile Modal — connectors open the ConnectionModal; no disconnect here
 * ────────────────────────────────────────────────────────────────────────── */
export default function ProfileModal({ me, connections = [], onClose }) {
  const navigate = useNavigate();

  /* ───────────── Tabs & toasts ───────────── */
  const [tab, setTab] = useState("profile");
  const [toast, setToast] = useState({ type: "", msg: "" });
  useEffect(() => {
    if (!toast.msg) return;
    const t = setTimeout(() => setToast({ type: "", msg: "" }), 3500);
    return () => clearTimeout(t);
  }, [toast]);
  const ok  = (msg) => setToast({ type: "ok",  msg });
  const err = (msg) => setToast({ type: "err", msg });

  /* ───────────── Profile fields ───────────── */
  const [displayName, setDisplayName] = useState(me?.name || "");
  const [avatarPreview, setAvatarPreview] = useState(me?.avatar_url || "");
  const [savingProfile, setSavingProfile] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/api/profile", { credentials: "include" });
        const j = await r.json();
        setDisplayName((j?.name || me?.name || "").trim());
        setAvatarPreview(j?.avatar_url || me?.avatar_url || "");
      } catch {}
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onAvatarChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAvatarPreview(URL.createObjectURL(file));
    try {
      const fd = new FormData();
      fd.append("avatar", file);
      const r = await fetch("/api/profile/avatar", {
        method: "POST", credentials: "include", body: fd,
      });
      if (!r.ok) throw new Error();
      ok("Avatar updated.");
      window.dispatchEvent(new Event("profile:changed"));
    } catch { err("Couldn't upload avatar. Try again."); }
  };

  const saveProfile = async () => {
    const name = (displayName || "").trim();
    if (!name) { err("Name cannot be empty."); return; }
    setSavingProfile(true);
    try {
      const r = await fetch("/api/profile", {
        method: "PATCH", credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!r.ok) throw new Error();
      ok("Profile saved.");
      window.dispatchEvent(new Event("profile:changed"));
    } catch { err("Couldn't save profile."); }
    finally { setSavingProfile(false); }
  };

  /* ───────────── Security / 2FA / API / Sessions ───────────── */
  const [pw, setPw] = useState({ current: "", next: "", confirm: "" });
  const [pwBusy, setPwBusy] = useState(false);
  const [pwShow, setPwShow] = useState(false);

  const changePassword = async () => {
    if (!pw.current || !pw.next || !pw.confirm) return err("Please complete all fields.");
    if (pw.next !== pw.confirm) return err("New passwords do not match.");
    setPwBusy(true);
    try {
      const r = await fetch("/api/security/change_password", {
        method: "POST", credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current_password: pw.current, new_password: pw.next }),
      });
      if (!r.ok) throw new Error();
      setPw({ current: "", next: "", confirm: "" });
      ok("Password changed.");
    } catch { err("Couldn't change password."); }
    finally { setPwBusy(false); }
  };

  const [twoFA, setTwoFA] = useState({ enabled:false, loading:true, qr:"", secret:"", otpauth_url:"", verifyCode:"" });
  const [twoFABusy, setTwoFABusy] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/api/security/2fa/status", { credentials: "include" });
        const j = await r.json().catch(() => ({}));
        setTwoFA((s) => ({ ...s, enabled: !!j.enabled, loading: false }));
      } catch { setTwoFA((s) => ({ ...s, loading: false })); }
    })();
  }, []);

  const start2FASetup = async () => {
    setTwoFABusy(true);
    try {
      const r = await fetch("/api/security/2fa/setup", { method: "POST", credentials: "include" });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !(j?.qr_svg || j?.otpauth_url)) throw new Error();
      setTwoFA((s) => ({ ...s, qr:j.qr_svg || "", secret:j.secret || "", otpauth_url:j.otpauth_url || "", enabled:false }));
      setTab("security");
      ok("Scan the QR with an authenticator, then enter the 6-digit code.");
    } catch { err("Couldn't start 2FA setup."); }
    finally { setTwoFABusy(false); }
  };
  const verify2FA = async () => {
    if (!twoFA.verifyCode) return;
    setTwoFABusy(true);
    try {
      const r = await fetch("/api/security/2fa/verify", {
        method: "POST", credentials: "include", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: twoFA.verifyCode }),
      });
      if (!r.ok) throw new Error();
      setTwoFA((s) => ({ ...s, enabled:true, qr:"", secret:"", otpauth_url:"", verifyCode:"" }));
      ok("Two-factor authentication enabled.");
    } catch { err("Invalid code. Try again."); }
    finally { setTwoFABusy(false); }
  };
  const disable2FA = async () => {
    setTwoFABusy(true);
    try {
      const r = await fetch("/api/security/2fa/disable", { method: "POST", credentials: "include" });
      if (!r.ok) throw new Error();
      setTwoFA((s) => ({ ...s, enabled:false, qr:"", secret:"", otpauth_url:"", verifyCode:"" }));
      ok("Two-factor authentication disabled.");
    } catch { err("Couldn't disable 2FA."); }
    finally { setTwoFABusy(false); }
  };

  const [token, setToken] = useState({ value:"•••• •••• •••• ••••", masked:true, loading:true, exists:false });
  const [tokenBusy, setTokenBusy] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [sessionsBusy, setSessionsBusy] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/api/profile/api_token", { credentials: "include" });
        const j = await r.json().catch(() => ({}));
        const has = !!j?.token || !!j?.exists;
        setToken({ value: has ? "•••• •••• •••• ••••" : "No token yet", masked:true, loading:false, exists:has });
      } catch {
        setToken({ value:"Unavailable", masked:true, loading:false, exists:false });
      }
      try {
        const r2 = await fetch("/api/auth/sessions", { credentials: "include" });
        const j2 = await r2.json().catch(() => ({}));
        setSessions(Array.isArray(j2?.sessions) ? j2.sessions : []);
      } catch {}
      finally { setSessionsBusy(false); }
    })();
  }, []);

  const revealToken = async () => {
    if (!token.masked) { setToken((t) => ({ ...t, masked:true })); return; }
    setTokenBusy(true);
    try {
      const r = await fetch("/api/profile/api_token", { credentials: "include" });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j?.token) throw new Error();
      setToken({ value:j.token, masked:false, loading:false, exists:true });
    } catch { err("Couldn't load token."); }
    finally { setTokenBusy(false); }
  };
  const copyToken   = async () => { try { await navigator.clipboard.writeText(token.value); ok("Token copied."); } catch { err("Clipboard blocked."); } };
  const rotateToken = async () => {
    setTokenBusy(true);
    try {
      const r = await fetch("/api/profile/api_token/rotate", { method: "POST", credentials: "include" });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j?.token) throw new Error();
      setToken({ value:j.token, masked:false, loading:false, exists:true });
      ok("Token rotated.");
    } catch { err("Couldn't rotate token."); }
    finally { setTokenBusy(false); }
  };
  const revokeOthers = async () => {
    setSessionsBusy(true);
    try {
      const r = await fetch("/api/auth/sessions/revoke_others", { method: "POST", credentials: "include" });
      if (!r.ok) throw new Error();
      const r2 = await fetch("/api/auth/sessions", { credentials: "include" });
      const j2 = await r2.json().catch(() => ({}));
      setSessions(Array.isArray(j2?.sessions) ? j2.sessions : []);
      ok("Signed out on other devices.");
    } catch { err("Couldn't revoke sessions."); }
    finally { setSessionsBusy(false); }
  };

  const [theme, setTheme] = useState(localStorage.getItem("theme") || "system");
  const [useAllSources, setUseAllSources] = useState(localStorage.getItem("pref.useAllSources") !== "0");
  useEffect(() => { localStorage.setItem("pref.useAllSources", useAllSources ? "1" : "0"); }, [useAllSources]);
  useEffect(() => {
    localStorage.setItem("theme", theme);
    const dark = theme === "dark" || (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", !!dark);
    window.dispatchEvent(new Event("theme:change"));
  }, [theme]);

  /* ───────────── Connections helpers ───────────── */
  const [connLocal, setConnLocal] = useState(connections);
  const byProvider = useMemo(() => {
    const m = {}; for (const c of (connLocal || [])) m[c.provider] = c; return m;
  }, [connLocal]);

  useEffect(() => { setConnLocal(connections || []); }, [connections]);

  // “Virtual” connectivity derived from MS/Google connections
  const msConnected =
    !!(byProvider.sharepoint?.connected || byProvider.onedrive?.connected ||
       byProvider.outlook?.connected || byProvider.outlook_calendar?.connected || byProvider.teams?.connected);
  const msEmail =
    byProvider.sharepoint?.account_email || byProvider.onedrive?.account_email ||
    byProvider.outlook?.account_email || byProvider.outlook_calendar?.account_email ||
    byProvider.teams?.account_email || "";

  const gConnected =
    !!(byProvider.google_drive?.connected || byProvider.gmail?.connected || byProvider.google_calendar?.connected);
  const gEmail =
    byProvider.google_drive?.account_email || byProvider.gmail?.account_email ||
    byProvider.google_calendar?.account_email || "";

  // Open connection modal globally, then close ProfileModal (avoid stacking)
  const openConn = (initial) => {
    window.dispatchEvent(new CustomEvent("connections:open", { detail: { initial } }));
    onClose?.();
  };

  // Openers
  const openSharePoint = () => openConn("sharepoint");
  const openOneDrive   = () => openConn("onedrive");
  const openGoogle     = () => openConn("google");
  const openDropbox    = () => openConn("dropbox");
  const openBox        = () => openConn("box");
  const openMega       = () => openConn("mega");
  const openTeams      = () => openConn("teams");
  const openSlack      = () => openConn("slack"); // ← NEW: Slack
  // New: mail & calendar
  const openOutlook          = () => openConn("outlook");
  const openOutlookCalendar  = () => openConn("outlook_calendar");
  const openGmail            = () => openConn("gmail");
  const openGoogleCalendar   = () => openConn("google_calendar");

  /* ───────────── UI helpers ───────────── */
  const Label = ({ children }) => <div className="text-xs font-semibold theme-text opacity-70">{children}</div>;
  const TabBtn = ({ icon, label, active, onClick }) => (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs transition ${
        active ? "bg-white text-black" : "border theme-border theme-text hover:bg-black/5"
      }`}
    >
      <span className="text-sm">{icon}</span>{label}
    </button>
  );
  const PwInput = ({ label, v, onChange, show }) => (
    <div>
      <div className="text-xs theme-text opacity-70">{label}</div>
      <input
        type={show ? "text" : "password"}
        value={v}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded-xl border theme-border theme-card theme-text px-3 py-2 text-sm outline-none focus:ring focus:ring-indigo-500/20"
        placeholder={label}
      />
    </div>
  );
  const Choice = ({ active, onClick, icon, label }) => (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm ${
        active ? "bg-white text-black" : "border theme-border theme-text hover:bg-black/5"
      }`}
    >
      <span>{icon}</span>{label}
    </button>
  );

  /* ───────────── Render ───────────── */
  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-[81] w-full max-w-3xl overflow-hidden rounded-2xl border theme-border theme-tint shadow-2xl backdrop-blur-xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b theme-border">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-full bg-gradient-to-br from-indigo-600 to-indigo-400 text-black font-bold">
              {(me?.name || me?.email || "U").charAt(0).toUpperCase()}
            </div>
            <div>
              <div className="text-sm font-semibold theme-text">{me?.name || "Your profile"}</div>
              <div className="text-xs theme-text opacity-70">{me?.email || "Unknown email"}</div>
            </div>
          </div>
          <button onClick={onClose} className="rounded-full p-2 hover:bg-black/5" aria-label="Close"><FiX className="theme-text" /></button>
        </div>

        {/* Tabs */}
        <div className="px-6 pt-3">
          <div className="flex flex-wrap gap-2">
            <TabBtn icon={<FiUser />}   label="Profile"        active={tab === "profile"}  onClick={() => setTab("profile")} />
            <TabBtn icon={<FiShield />} label="Security"       active={tab === "security"} onClick={() => setTab("security")} />
            <TabBtn icon={<FiSun />}    label="Preferences"    active={tab === "prefs"}    onClick={() => setTab("prefs")} />
            <TabBtn icon={<FiKey />}    label="API & Sessions" active={tab === "api"}      onClick={() => setTab("api")} />
          </div>
        </div>

        {/* Body */}
        <div className="max-h-[70vh] overflow-y-auto px-6 py-4 theme-text">
          {tab === "profile" && (
            <section className="space-y-6">
              {/* Basic profile */}
              <div className="flex items-start gap-4">
                <div className="relative">
                  <div className="h-20 w-20 rounded-full ring-2 theme-border overflow-hidden theme-tint">
                    {avatarPreview ? <img src={avatarPreview} alt="avatar" className="h-full w-full object-cover" /> : null}
                  </div>
                  <label className="absolute -bottom-1 -right-1 grid h-8 w-8 place-items-center rounded-full bg-white text-black shadow cursor-pointer">
                    <FiEdit2 />
                    <input type="file" accept="image/*" className="hidden" onChange={onAvatarChange} />
                  </label>
                </div>
                <div className="flex-1">
                  <Label>Display name</Label>
                  <input
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    className="mt-1 w-full rounded-xl border theme-border theme-card theme-text px-3 py-2 text-sm outline-none focus:ring focus:ring-indigo-500/20"
                    placeholder="Your name"
                  />
                  <div className="mt-3 flex gap-2">
                    <button
                      onClick={saveProfile}
                      disabled={savingProfile || !displayName.trim()}
                      className="rounded-xl bg-white px-4 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50"
                    >
                      {savingProfile ? "Saving…" : "Save changes"}
                    </button>
                    <button
                      onClick={() => { navigator.clipboard.writeText(me?.id || ""); ok("User ID copied."); }}
                      className="inline-flex items-center gap-2 rounded-xl border theme-border px-3 py-2 text-sm hover:bg-black/5"
                    >
                      <FiCopy className="opacity-80" /> Copy user ID
                    </button>
                  </div>
                </div>
              </div>

              {/* Connections manager */}
              <div className="pt-2">
                <Label>Connections</Label>

                <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-2">
                  <ProviderRow
                    icon={<FaMicrosoft className="text-teal-300" />}
                    name="SharePoint"
                    connected={!!byProvider.sharepoint?.connected}
                    email={byProvider.sharepoint?.account_email}
                    onOpen={openSharePoint}
                  />
                  <ProviderRow
                    icon={<FiCloud className="text-sky-300" />}
                    name="OneDrive"
                    connected={!!byProvider.onedrive?.connected}
                    email={byProvider.onedrive?.account_email}
                    onOpen={openOneDrive}
                  />
                  <ProviderRow
                    icon={<PiMicrosoftTeamsLogoFill className="text-purple-300" />}
                    name="Microsoft Teams"
                    connected={!!byProvider.teams?.connected}
                    email={byProvider.teams?.account_email}
                    onOpen={openTeams}
                    helpText="Search & summarize Teams messages and files."
                  />
                  {/* NEW: Slack */}
                  <ProviderRow
                    icon={<FaSlack className="text-fuchsia-300" />}
                    name="Slack"
                    connected={!!byProvider.slack?.connected}
                    email={byProvider.slack?.account_email /* team name */}
                    onOpen={openSlack}
                    helpText="Search Slack messages with links & timestamps."
                  />
                  <ProviderRow
                    icon={<SiGoogledrive className="text-emerald-300" />}
                    name="Google Drive"
                    connected={!!byProvider.google_drive?.connected}
                    email={byProvider.google_drive?.account_email}
                    onOpen={openGoogle}
                  />
                  <ProviderRow
                    icon={<SiDropbox className="text-sky-400" />}
                    name="Dropbox"
                    connected={!!byProvider.dropbox?.connected}
                    email={byProvider.dropbox?.account_email}
                    onOpen={openDropbox}
                  />
                  <ProviderRow
                    icon={<SiBox className="text-blue-400" />}
                    name="Box"
                    connected={!!byProvider.box?.connected}
                    email={byProvider.box?.account_email}
                    onOpen={openBox}
                  />
                  <ProviderRow
                    icon={<SiMega className="text-rose-300" />}
                    name="MEGA"
                    connected={!!byProvider.mega?.connected}
                    email={byProvider.mega?.account_email}
                    onOpen={openMega}
                    helpText="Advanced setup in modal (session & links)."
                  />
                </div>

                {/* Mail & Calendar */}
                <div className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-2">
                  <ProviderRow
                    icon={<FiMail className="text-sky-300" />}
                    name="Outlook Mail"
                    connected={msConnected}
                    email={msEmail}
                    onOpen={openOutlook}
                  />
                  <ProviderRow
                    icon={<FiCalendar className="text-sky-300" />}
                    name="Outlook Calendar"
                    connected={msConnected}
                    email={msEmail}
                    onOpen={openOutlookCalendar}
                  />
                  <ProviderRow
                    icon={<FiMail className="text-emerald-300" />}
                    name="Gmail"
                    connected={gConnected}
                    email={gEmail}
                    onOpen={openGmail}
                  />
                  <ProviderRow
                    icon={<FiCalendar className="text-emerald-300" />}
                    name="Google Calendar"
                    connected={gConnected}
                    email={gEmail}
                    onOpen={openGoogleCalendar}
                  />
                </div>

                <div className="mt-3 text-xs theme-text opacity-70">
                  Tip: You can also manage connections on the <a href="/dashboard" className="underline">Dashboard</a>.
                </div>
              </div>
            </section>
          )}

          {tab === "security" && (
            <section className="space-y-6">
              <div>
                <Label>Change password</Label>
                <div className="mt-2 grid gap-2 md:grid-cols-3">
                  <PwInput label="Current" v={pw.current} onChange={(v) => setPw((s) => ({ ...s, current: v }))} show={pwShow} />
                  <PwInput label="New"     v={pw.next}    onChange={(v) => setPw((s) => ({ ...s, next: v }))}    show={pwShow} />
                  <PwInput label="Confirm" v={pw.confirm} onChange={(v) => setPw((s) => ({ ...s, confirm: v }))} show={pwShow} />
                </div>
                <div className="mt-2 flex items-center gap-3">
                  <button onClick={() => setPwShow((s) => !s)} className="rounded-xl border theme-border px-3 py-2 text-sm hover:bg-black/5">
                    {pwShow ? (<><FiEyeOff className="inline mr-1" /> Hide</>) : (<><FiEye className="inline mr-1" /> Show</>)}
                  </button>
                  <button
                    onClick={changePassword}
                    disabled={pwBusy}
                    className="rounded-xl bg-white px-4 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50"
                  >
                    {pwBusy ? "Saving…" : "Update password"}
                  </button>
                </div>
              </div>

              {/* 2FA */}
              <div>
                <Label>Two-factor authentication</Label>
                <div className="mt-2 rounded-xl border theme-border p-3">
                  {twoFA.loading ? (
                    <div className="text-sm theme-text opacity-70">Loading 2FA status…</div>
                  ) : twoFA.enabled ? (
                    <div className="flex items-center justify-between">
                      <div className="text-sm theme-text">
                        2FA is <span className="text-emerald-600">enabled</span>.
                      </div>
                      <button
                        onClick={disable2FA}
                        disabled={twoFABusy}
                        className="rounded-xl px-3 py-2 text-sm theme-text hover:bg-black/5 disabled:opacity-50 border theme-border"
                      >
                        Disable 2FA
                      </button>
                    </div>
                  ) : !twoFA.qr ? (
                    <div className="flex items-center justify-between">
                      <div className="text-sm theme-text">Protect your account with an authenticator app.</div>
                      <button
                        onClick={start2FASetup}
                        disabled={twoFABusy}
                        className="rounded-xl bg-white px-3 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50"
                      >
                        {twoFABusy ? "Working…" : "Enable 2FA"}
                      </button>
                    </div>
                  ) : (
                    <TwoFAStep twoFA={twoFA} setTwoFA={setTwoFA} verify2FA={verify2FA} twoFABusy={twoFABusy} />
                  )}
                </div>
              </div>

              <DangerZone onExport={async () => {
                try {
                  const r = await fetch("/api/account/export", { credentials: "include" });
                  const blob = await r.blob();
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url; a.download = "export.zip"; a.click();
                  URL.revokeObjectURL(url);
                  ok("Export started.");
                } catch { err("Export not available."); }
              }} onDelete={async () => {
                const confirmStr = prompt("Type DELETE to permanently delete your account.");
                if (confirmStr !== "DELETE") return;
                try {
                  const r = await fetch("/api/account", { method: "DELETE", credentials: "include" });
                  if (!r.ok) throw new Error();
                  ok("Account deleted."); setTimeout(() => navigate("/login", { replace: true }), 800);
                } catch { err("Couldn't delete account."); }
              }} />
            </section>
          )}

          {tab === "prefs" && (
            <section className="space-y-5">
              <div>
                <Label>Theme</Label>
                <div className="mt-2 flex flex-wrap gap-2">
                  <Choice active={theme === "system"} onClick={() => setTheme("system")} icon={<FiMonitor />} label="System" />
                  <Choice active={theme === "light"}  onClick={() => setTheme("light")}  icon={<FiSun />}     label="Light" />
                  <Choice active={theme === "dark"}   onClick={() => setTheme("dark")}   icon={<FiMoon />}    label="Dark" />
                </div>
              </div>
              <div>
                <Label>Search defaults</Label>
                <div className="mt-2 flex items-center justify-between rounded-xl border theme-border p-3">
                  <div className="text-sm theme-text">Use all connected sources by default</div>
                  <ToggleSwitch
                    checked={localStorage.getItem("pref.useAllSources") !== "0"}
                    onChange={(v) => localStorage.setItem("pref.useAllSources", v ? "1" : "0")}
                  />
                </div>
              </div>
            </section>
          )}

          {tab === "api" && (
            <section className="space-y-6">
              <div className="rounded-xl border theme-border p-3">
                <Label>API token</Label>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <code className="rounded-lg theme-tint px-3 py-2 text-sm theme-text">{token.value}</code>
                  <button onClick={revealToken} disabled={tokenBusy} className="rounded-xl border theme-border px-3 py-2 text-sm hover:bg-black/5">
                    {token.masked ? (<><FiEye className="inline mr-1" /> Reveal</>) : (<><FiEyeOff className="inline mr-1" /> Hide</>)}
                  </button>
                  <button onClick={copyToken} disabled={token.masked || tokenBusy} className="rounded-xl border theme-border px-3 py-2 text-sm hover:bg-black/5 disabled:opacity-50">
                    <FiCopy className="inline mr-1" /> Copy
                  </button>
                  <button onClick={rotateToken} disabled={tokenBusy} className="rounded-xl bg-white px-3 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50">
                    Rotate
                  </button>
                </div>
                {!token.exists && <div className="mt-2 text-xs theme-text opacity-70">No token yet — click <b>Rotate</b> to create one.</div>}
              </div>

              <div className="rounded-xl border theme-border p-3">
                <div className="mb-2 flex items-center justify-between">
                  <Label>Active sessions</Label>
                  <button onClick={revokeOthers} disabled={sessionsBusy} className="rounded-xl bg-white px-3 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50">
                    Sign out others
                  </button>
                </div>
                {sessionsBusy ? (
                  <div className="text-sm theme-text opacity-70">Loading sessions…</div>
                ) : sessions.length === 0 ? (
                  <div className="text-sm theme-text opacity-70">Only this device is signed in.</div>
                ) : (
                  <ul className="divide-y theme-border">
                    {sessions.map((s) => (
                      <li key={s.id} className="flex items-center justify-between py-2 text-sm">
                        <div className="flex items-center gap-2 theme-text">
                          <FiSmartphone className="opacity-80" />
                          <span className="truncate max-w-[14rem]">{s.user_agent || "Device"}</span>
                        </div>
                        <div className="theme-text opacity-70">Last active: {s.last_active ? new Date(s.last_active).toLocaleString() : "—"}</div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </section>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 border-t theme-border px-6 py-3">
          <button onClick={onClose} className="rounded-xl px-4 py-2 text-sm font-semibold theme-text hover:bg-black/5 theme-card border theme-border">
            Close
          </button>
        </div>
      </div>

      {/* Toast */}
      {toast.msg && (
        <div
          className={`fixed bottom-6 left-1/2 z-[90] -translate-x-1/2 rounded-xl border px-4 py-3 text-sm shadow-2xl ${
            toast.type === "ok"
              ? "bg-emerald-500/10 border-emerald-400/30 text-emerald-200"
              : "bg-red-500/10 border-red-400/30 text-red-200"
          }`}
        >
          {toast.msg}
        </div>
      )}
    </div>
  );
}

/* ───────────────────────────── Sub-components ─────────────────────────── */
function ProviderRow({ icon, name, connected, email, onOpen, helpText }) {
  return (
    <div
      className="flex items-center justify-between rounded-xl border theme-border theme-card px-3 py-2 cursor-pointer"
      onClick={onOpen}
      title={connected ? "Manage connection" : "Connect"}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span className="w-5 h-5 grid place-items-center">{icon}</span>
        <div className="min-w-0">
          <div className="text-sm theme-text">{name}</div>
          <div className="text-[11px] theme-text opacity-70 truncate">
            {email ? `Connected as ${email}` : (connected ? "Connected" : "Not connected")}
          </div>
          {helpText && <div className="text-[10px] theme-text opacity-60">{helpText}</div>}
        </div>
      </div>
      <span className={`h-2.5 w-2.5 rounded-full ${connected ? "bg-emerald-500" : "bg-zinc-500"}`} />
    </div>
  );
}

function TwoFAStep({ twoFA, setTwoFA, verify2FA, twoFABusy }) {
  return (
    <div className="grid gap-3 md:grid-cols-[minmax(160px,220px),1fr]">
      <div
        className="w-[160px] md:w-[220px] shrink-0 rounded-xl bg-white p-2 overflow-hidden
                   [&>svg]:block [&>svg]:w-full [&>svg]:h-auto [&>svg]:max-w-full"
        dangerouslySetInnerHTML={{ __html: twoFA.qr }}
      />
      <div>
        <div className="text-sm theme-text opacity-80">Scan the QR, then enter the 6-digit code to activate.</div>
        <div className="mt-2 text-xs theme-text opacity-70 break-words">
          Secret: <code className="px-1 py-0.5 rounded theme-tint">{twoFA.secret || "—"}</code>
          <button onClick={() => navigator.clipboard.writeText(twoFA.secret || "")}
                  className="ml-2 underline hover:opacity-80">Copy secret</button>
        </div>
        <div className="mt-1 text-xs theme-text opacity-70 break-words">
          otpauth: <span className="break-all">{twoFA.otpauth_url || "—"}</span>
          <button onClick={() => navigator.clipboard.writeText(twoFA.otpauth_url || "")}
                  className="ml-2 underline hover:opacity-80">Copy link</button>
        </div>
        <input
          value={twoFA.verifyCode}
          onChange={(e) =>
            setTwoFA((s) => ({ ...s, verifyCode: e.target.value.replace(/\D/g, "").slice(0, 6) }))
          }
          inputMode="numeric" pattern="[0-9]*"
          className="mt-3 w-full rounded-xl border theme-border theme-card theme-text px-3 py-2 text-sm outline-none focus:ring focus:ring-indigo-500/20"
          placeholder="123456"
        />
        <div className="mt-2">
          <button
            onClick={verify2FA}
            disabled={twoFABusy || twoFA.verifyCode.length !== 6}
            className="rounded-xl bg-white px-3 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50"
          >
            {twoFABusy ? "Verifying…" : "Verify & Enable"}
          </button>
        </div>
      </div>
    </div>
  );
}

function ToggleSwitch({ checked, onChange }) {
  return (
    <label className="relative inline-flex cursor-pointer items-center">
      <input type="checkbox" className="peer sr-only" checked={!!checked} onChange={(e) => onChange(e.target.checked)} />
      <div className="peer h-5 w-9 rounded-full bg-white/70 after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:bg-white after:transition-all peer-checked:bg-emerald-500 peer-checked:after:translate-x-4" />
    </label>
  );
}

function DangerZone({ onExport, onDelete }) {
  return (
    <div className="rounded-xl border border-red-500/30 theme-card p-3">
      <div className="mb-2 text-xs font-semibold text-red-600">Danger zone</div>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={onExport}
          className="inline-flex items-center gap-2 rounded-xl border border-red-500/30 px-3 py-2 text-sm text-red-700 hover:bg-red-50"
        >
          <FiDownload /> Export my data
        </button>
        <button
          onClick={onDelete}
          className="inline-flex items-center gap-2 rounded-xl bg-red-600 px-3 py-2 text-sm font-semibold text-white hover:bg-red-500"
        >
          <FiTrash2 /> Delete account
        </button>
      </div>
    </div>
  );
}
