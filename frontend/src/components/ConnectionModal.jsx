// src/components/ConnectionModal.jsx
import React, { useEffect, useMemo, useState } from "react";

/**
 * Minimal, provider-focused Connections Modal
 * Supports: sharepoint, onedrive, outlook, outlook_calendar, teams,
 *           google (Drive), gmail, google_calendar, dropbox, box, mega, slack
 *
 * Props:
 *   open: boolean
 *   onClose: () => void
 *   initial: provider key from the list above
 */
export default function ConnectionModal({ open, onClose, initial = "google" }) {
  const provider = normalize(initial);

  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [meEmail, setMeEmail] = useState("");
  const [connections, setConnections] = useState([]);

  // MEGA local state
  const [megaSession, setMegaSession] = useState("");
  const [megaLink, setMegaLink] = useState("");
  const [megaBusy, setMegaBusy] = useState(false);

  const meta = useMemo(() => metaFor(provider), [provider]);

  useEffect(() => {
    if (!open) return;
    setError(""); setInfo(""); setBusy(false);

    (async () => {
      try {
        setLoading(true);
        const u = await fetch("/check_login", { credentials: "include", cache: "no-store" });
        const uj = await u.json().catch(() => ({}));
        if (uj?.user_email) setMeEmail(uj.user_email);

        const r = await fetch("/api/connections", {
          credentials: "include", cache: "no-store", headers: { "Cache-Control": "no-cache" },
        });
        const j = await r.json().catch(() => ({}));
        setConnections(Array.isArray(j?.connections) ? j.connections : []);
      } catch {
        setError("Failed to load connection status.");
      } finally {
        setLoading(false);
      }
    })();

    // show friendly success chip when redirected back
    try {
      const params = new URLSearchParams(window.location.search);
      const c = params.get("connected");
      if (c) {
        setInfo(successMessage(c));
        const url = new URL(window.location.href);
        url.searchParams.delete("connected");
        window.history.replaceState({}, "", url.toString());
      }
    } catch {}
  }, [open, provider]);

  // helpers
  const get = (name) => connections.find((c) => String(c.provider).toLowerCase() === name);

  // Microsoft family
  const sp = get("sharepoint");
  const od = get("onedrive");
  const om = get("outlook");
  const oc = get("outlook_calendar");
  const tm = get("teams");
  const msConnected = !!(sp?.connected || od?.connected || om?.connected || oc?.connected);
  const msEmail = sp?.account_email || od?.account_email || om?.account_email || oc?.account_email || tm?.account_email || "";

  // Google family
  const gd   = get("google_drive") || get("google");
  const gmail = get("gmail");
  const gcal  = get("google_calendar");
  const gConnected = !!(gd?.connected || gmail?.connected || gcal?.connected);
  const gEmail = gd?.account_email || gmail?.account_email || gcal?.account_email || "";

  // others
  const dbx = get("dropbox");
  const bx  = get("box");
  const mg  = get("mega");
  const sk  = get("slack");

  const isConnected =
    provider === "sharepoint"        ? !!sp?.connected :
    provider === "onedrive"          ? !!od?.connected :
    provider === "outlook"           ? !!om?.connected :
    provider === "outlook_calendar"  ? !!oc?.connected :
    provider === "teams"             ? !!tm?.connected :
    provider === "google"            ? !!gd?.connected :
    provider === "gmail"             ? !!gmail?.connected :
    provider === "google_calendar"   ? !!gcal?.connected :
    provider === "dropbox"           ? !!dbx?.connected :
    provider === "box"               ? !!bx?.connected :
    provider === "mega"              ? !!mg?.connected :
    provider === "slack"             ? !!sk?.connected :
    false;

  const connectedEmail =
    provider === "teams" ? (tm?.account_email || msEmail || meEmail) :
    provider.startsWith("outlook") ? (msEmail || meEmail) :
    (provider === "google" || provider === "gmail" || provider === "google_calendar") ? (gEmail || meEmail) :
    (get(provider)?.account_email || meEmail);

  /* ----------------- Connect flows ----------------- */
  async function connectMicrosoft() { await goOAuth("/api/connections/microsoft/authurl"); }
  async function connectTeams()     { await goOAuth("/api/connections/microsoft/authurl"); }
  async function connectGoogle()    { await goOAuth("/api/connections/google/authurl"); }
  async function connectDropbox()   { await goOAuth("/api/connections/dropbox/authurl"); }
  async function connectBox()       { await goOAuth("/api/connections/box/authurl"); }
  async function connectSlack()     { await goOAuth("/api/connections/slack/start"); } // Slack uses {redirect}

  async function goOAuth(url) {
    setError(""); setInfo("");
    try {
      setBusy(true);
      const r = await fetch(url, { credentials: "include", cache: "no-store" });
      const text = await r.text();
      let j;
      try { j = JSON.parse(text); }
      catch {
        if (!r.ok) throw new Error(text || `Auth failed (${r.status})`);
        throw new Error("Unexpected auth response.");
      }
      if (!r.ok) throw new Error(j?.error || "Auth failed.");
      const authUrl = j?.auth_url || j?.redirect; // support both shapes
      if (!authUrl) throw new Error("Missing authorization URL.");
      window.location.href = authUrl;
    } catch (e) {
      setError(e.message || "Could not start authentication.");
    } finally {
      setBusy(false);
    }
  }

  /* --------------- Disconnect flows --------------- */
  async function disconnectMicrosoft(target /* 'sharepoint' | 'onedrive' | 'outlook' | 'outlook_calendar' */) {
    const safeTarget = ["sharepoint","onedrive","outlook","outlook_calendar"].includes(target) ? target : "onedrive";
    await doDisconnect(`/api/connections/microsoft/disconnect/${safeTarget}`, `Microsoft disconnected.`);
  }
  async function disconnectTeams()  { await doDisconnect("/api/connections/microsoft/disconnect/teams",  "Microsoft Teams disconnected."); }
  async function disconnectGoogle() { await doDisconnect("/api/connections/google/disconnect", "Google disconnected."); }
  async function disconnectDropbox(){ await doDisconnect("/api/connections/dropbox/disconnect","Dropbox disconnected."); }
  async function disconnectBox()    { await doDisconnect("/api/connections/box/disconnect",    "Box disconnected."); }
  async function disconnectMega()   { await doDisconnect("/api/connections/mega/disconnect",   "MEGA disconnected."); }
  async function disconnectSlack()  { await doDisconnect("/api/connections/slack/disconnect",  "Slack disconnected."); }

  async function doDisconnect(url, okMsg) {
    setError(""); setInfo("");
    try {
      setBusy(true);
      const r = await fetch(url, { method: "DELETE", credentials: "include", cache: "no-store" });
      await r.json().catch(() => ({}));
      setInfo(okMsg);
      await refreshConnections();
    } catch {
      setError("Failed to disconnect.");
    } finally {
      setBusy(false);
    }
  }

  async function refreshConnections() {
    try {
      const r = await fetch("/api/connections", {
        credentials: "include", cache: "no-store", headers: { "Cache-Control": "no-cache" },
      });
      const j = await r.json().catch(() => ({}));
      setConnections(Array.isArray(j?.connections) ? j.connections : []);
    } catch {}
  }

  /* ------------------ MEGA helpers ------------------ */
  async function megaSaveSession() {
    if (!megaSession.trim()) return;
    setError(""); setInfo(""); setMegaBusy(true);
    try {
      const r = await fetch("/api/connections/mega/connect", {
        method: "POST", credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: megaSession }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j?.error || "Could not save session.");
      setMegaSession("");
      setInfo("MEGA session saved.");
      await refreshConnections();
    } catch (e) {
      setError(e.message || "MEGA error.");
    } finally {
      setMegaBusy(false);
    }
  }
  async function megaAddLink() {
    if (!megaLink.trim()) return;
    setError(""); setInfo(""); setMegaBusy(true);
    try {
      const r = await fetch("/api/connections/mega/add_link", {
        method: "POST", credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ link: megaLink }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j?.error || "Could not add link.");
      setMegaLink("");
      setInfo("MEGA link added.");
      await refreshConnections();
    } catch (e) {
      setError(e.message || "MEGA error.");
    } finally {
      setMegaBusy(false);
    }
  }

  if (!open) return null;

  const isMs = ["sharepoint", "onedrive", "outlook", "outlook_calendar", "teams"].includes(provider);
  const isGoogle = ["google", "gmail", "google_calendar"].includes(provider);

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center" role="dialog" aria-modal="true">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Card */}
      <div className="relative z-[71] w-full max-w-md md:max-w-lg">
        <div className="overflow-hidden rounded-2xl border border-white/10 bg-[#0b1324]/95 shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
            <div className="flex items-center gap-3">
              <span className="grid h-9 w-9 place-items-center rounded-lg bg-white/10 ring-1 ring-white/10">
                {meta.icon}
              </span>
              <div>
                <div className="text-[15px] font-semibold text-white">
                  {isConnected ? `Manage ${meta.title}` : `Connect ${meta.title}`}
                </div>
                <div className="text-xs text-white/60">{meta.subtitle}</div>
              </div>
            </div>
            <button
              onClick={onClose}
              className="rounded-full p-2 text-white/70 hover:text-white hover:bg-white/5"
              aria-label="Close"
            >
              ✕
            </button>
          </div>

          {/* Body */}
          <div className="p-5">
            {error && (
              <div className="mb-3 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
                {error}
              </div>
            )}
            {info && (
              <div className="mb-3 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-200">
                {info}
              </div>
            )}

            {loading ? (
              <div className="h-16 animate-pulse rounded-md bg-white/5" />
            ) : provider === "mega" ? (
              <MegaSection
                connected={!!mg?.connected}
                email={mg?.account_email}
                session={megaSession}
                setSession={setMegaSession}
                link={megaLink}
                setLink={setMegaLink}
                onSaveSession={megaSaveSession}
                onAddLink={megaAddLink}
                onDisconnect={disconnectMega}
                busy={megaBusy}
              />
            ) : (
              <OAuthSection
                provider={provider}
                connected={isConnected}
                email={connectedEmail}
                busy={busy}
                onConnect={
                  provider === "teams"   ? connectTeams   :
                  provider === "slack"   ? connectSlack   :
                  isGoogle               ? connectGoogle  :
                  provider === "dropbox" ? connectDropbox :
                  provider === "box"     ? connectBox     :
                  /* Microsoft family (SP/OD/Outlook/OutlookCal) */
                  connectMicrosoft
                }
                onDisconnect={
                  provider === "teams"   ? disconnectTeams  :
                  provider === "slack"   ? disconnectSlack  :
                  isGoogle               ? disconnectGoogle :
                  provider === "dropbox" ? disconnectDropbox:
                  provider === "box"     ? disconnectBox    :
                  /* Microsoft family -> choose a valid target for your API */
                  () => disconnectMicrosoft(
                    provider === "sharepoint" ? "sharepoint" :
                    provider === "onedrive"   ? "onedrive"   :
                    provider === "outlook"    ? "outlook"    : "outlook_calendar"
                  )
                }
                msNote={isMs}
              />
            )}

            {/* Footer (tiny) */}
            <div className="mt-5 flex items-center justify-between text-[11px] text-white/60">
              <span>Signed in as <b className="text-white/80">{meEmail || "you"}</b></span>
              <div className="flex items-center gap-1">
                <Pill>Secure by design</Pill>
                <Pill>OAuth 2.0</Pill>
                <Pill>Read-only</Pill>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* -------------------- Sections -------------------- */

function OAuthSection({ provider, connected, email, busy, onConnect, onDisconnect, msNote }) {
  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
        <div className="text-sm text-white">
          {connected ? (
            <>
              <span className="inline-block rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] text-emerald-300 mr-2">Connected</span>
              {email && <span className="text-white/70">as <b className="text-white/90">{email}</b></span>}
            </>
          ) : (
            <span className="text-white/70">Not connected</span>
          )}
        </div>
        {msNote && (
          <div className="mt-1 text-[11px] text-white/50">
            Powered by your Microsoft account (SharePoint / OneDrive / Outlook / Teams).
          </div>
        )}

        <div className="mt-3 flex gap-2">
          {connected ? (
            <button
              onClick={onDisconnect}
              disabled={busy}
              className="rounded-md border border-white/15 bg-white/10 px-3 py-2 text-sm text-white hover:bg-white/15 disabled:opacity-50"
            >
              Disconnect
            </button>
          ) : (
            <button
              onClick={onConnect}
              disabled={busy}
              className="rounded-md bg-white px-3 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50"
            >
              {busy ? "Opening…" : "Connect"}
            </button>
          )}
        </div>
      </div>

      <ul className="list-disc pl-5 text-[11px] text-white/60 space-y-1">
        <li>You’ll be redirected to the provider’s consent screen.</li>
        <li>We never store your password.</li>
        <li>Access is read-only by default.</li>
      </ul>
    </div>
  );
}

function MegaSection({ connected, email, session, setSession, link, setLink, onSaveSession, onAddLink, onDisconnect, busy }) {
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
        <div className="text-sm text-white">
          {connected ? (
            <>
              <span className="inline-block rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] text-emerald-300 mr-2">Connected</span>
              {email && <span className="text-white/70">as <b className="text-white/90">{email}</b></span>}
            </>
          ) : (
            <span className="text-white/70">Not connected</span>
          )}
        </div>

        <textarea
          rows={3}
          value={session}
          onChange={(e) => setSession(e.target.value)}
          className="mt-3 w-full rounded-md border border-white/10 bg-transparent px-3 py-2 text-sm text-white outline-none placeholder:text-white/30"
          placeholder="Paste MEGAcmd session (mega-executablesession)…"
        />
        <div className="mt-2 flex gap-2">
          <button
            onClick={onSaveSession}
            disabled={busy || !session.trim()}
            className="rounded-md bg-white px-3 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50"
          >
            {busy ? "Saving…" : "Save session"}
          </button>
          {connected && (
            <button
              onClick={onDisconnect}
              disabled={busy}
              className="rounded-md border border-white/15 bg-white/10 px-3 py-2 text-sm text-white hover:bg-white/15 disabled:opacity-50"
            >
              Disconnect
            </button>
          )}
        </div>
      </div>

      <div className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
        <div className="text-sm text-white">Add public link</div>
        <div className="mt-1 text-[11px] text-white/60">
          Supports file and folder links: <code>https://mega.nz/file/...#KEY</code> or <code>https://mega.nz/folder/...#KEY</code>.
        </div>
        <input
          value={link}
          onChange={(e) => setLink(e.target.value)}
          className="mt-2 w-full rounded-md border border-white/10 bg-transparent px-3 py-2 text-sm text-white outline-none placeholder:text-white/30"
          placeholder="https://mega.nz/file/…#…"
        />
        <div className="mt-2">
          <button
            onClick={onAddLink}
            disabled={busy || !link.trim()}
            className="rounded-md bg-white px-3 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50"
          >
            {busy ? "Working…" : "Add link"}
          </button>
        </div>
      </div>
    </div>
  );
}

/* -------------------- Tiny UI helpers -------------------- */

function Pill({ children }) {
  return (
    <span className="rounded-full border border-white/15 bg-white/10 px-2 py-[2px] text-[10px] text-white/70">
      {children}
    </span>
  );
}

/* -------------------- Meta & Icons -------------------- */

function normalize(key) {
  const k = String(key || "google").toLowerCase().replace(/\s+/g, "_");
  const allowed = [
    "sharepoint", "onedrive", "outlook", "outlook_calendar", "teams",
    "google", "gmail", "google_calendar",
    "dropbox", "box", "mega", "slack"
  ];
  return allowed.includes(k) ? k : "google";
}
function titleFor(key) {
  return key === "sharepoint"       ? "SharePoint"
       : key === "onedrive"         ? "OneDrive"
       : key === "outlook"          ? "Outlook Mail"
       : key === "outlook_calendar" ? "Outlook Calendar"
       : key === "teams"            ? "Microsoft Teams"
       : key === "google"           ? "Google Drive"
       : key === "gmail"            ? "Gmail"
       : key === "google_calendar"  ? "Google Calendar"
       : key === "dropbox"          ? "Dropbox"
       : key === "box"              ? "Box"
       : key === "mega"             ? "MEGA"
       : key === "slack"            ? "Slack"
       : "Connect";
}
function metaFor(key) {
  const base = {
    sharepoint:       { title: "SharePoint",       subtitle: "Search your SharePoint content via Microsoft OAuth.",             icon: <MicrosoftLogo /> },
    onedrive:         { title: "OneDrive",         subtitle: "Uses the same Microsoft connection as SharePoint.",               icon: <OneDriveLogo /> },
    outlook:          { title: "Outlook Mail",     subtitle: "Read Outlook email via your Microsoft account (read-only).",     icon: <MicrosoftLogo /> },
    outlook_calendar: { title: "Outlook Calendar", subtitle: "Read Outlook calendar via your Microsoft account (read-only).",  icon: <MicrosoftLogo /> },
    teams:            { title: "Microsoft Teams",  subtitle: "Read Teams chats/channels via your Microsoft account (read-only).", icon: <TeamsLogo /> },
    google:           { title: "Google Drive",     subtitle: "Connect with Google OAuth (read-only).",                          icon: <GoogleLogo /> },
    gmail:            { title: "Gmail",            subtitle: "Read Gmail via your Google account (read-only).",                 icon: <GoogleLogo /> },
    google_calendar:  { title: "Google Calendar",  subtitle: "Read Google Calendar via your Google account (read-only).",       icon: <GoogleLogo /> },
    dropbox:          { title: "Dropbox",          subtitle: "Search across your Dropbox folders and files.",                   icon: <DropboxLogo /> },
    box:              { title: "Box",              subtitle: "Connect with Box OAuth (read-only).",                             icon: <BoxLogo /> },
    mega:             { title: "MEGA",             subtitle: "Add a session string and paste public links.",                    icon: <MegaLogo /> },
    slack:            { title: "Slack",            subtitle: "Search Slack messages with links & timestamps (read-only).",      icon: <SlackLogo /> },
  };
  return base[key] || base.google;
}
function successMessage(code) {
  return code === "ms"      ? "Microsoft connected."
       : code === "teams"   ? "Microsoft Teams connected."
       : code === "google"  ? "Google connected."
       : code === "dropbox" ? "Dropbox connected."
       : code === "box"     ? "Box connected."
       : code === "mega"    ? "MEGA updated."
       : code === "slack"   ? "Slack connected."
       : "Connected.";
}

/* simple glyphs (no external icons needed) */
function MicrosoftLogo() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#f35325" d="M23 23H4V4h19z" />
      <path fill="#81bc06" d="M44 23H25V4h19z" />
      <path fill="#05a6f0" d="M23 44H4V25h19z" />
      <path fill="#ffba08" d="M44 44H25V25h19z" />
    </svg>
  );
}
function OneDriveLogo() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#1976D2" d="M24 8a9 9 0 00-8.2 5.3C12.5 13 9 16.6 9 21c0 4.4 3.6 8 8 8h20a6 6 0 000-12c-.4 0-.8.1-1.2.1A9 9 0 0024 8z" />
    </svg>
  );
}
function GoogleLogo() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.8 31.7 29.4 35 24 35c-6.6 0-12-5.4-12-12s5.4-12 12-12c3 0 5.7 1.1 7.8 3l5.7-5.7C33.6 5.1 29 3 24 3 12.9 3 4 11.9 4 23s8.9 20 20 20c10 0 18.5-7.3 19.9-16.8.1-.8.1-1.6.1-2.7z" />
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.4 16.1 18.8 13 24 13c3 0 5.7 1.1 7.8 3l5.7-5.7C33.6 5.1 29 3 24 3 16.4 3 9.8 7.1 6.3 14.7z" />
      <path fill="#4CAF50" d="M24 43c5.3 0 9.9-1.7 13.5-4.7l-6.2-4.9C29.1 35.3 26.7 36 24 36c-5.3 0-9.7-3.3-11.4-8l-6.5 5C9.6 39 16.3 43 24 43z" />
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-1.3 3.8-5.7 7-11.3 7-6.6 0-12-5.4-12-12 0-1 .1-2 .3-3h-.1l-6.6-4.8C4.6 17.2 4 20 4 23c0 11.1 8.9 20 20 20 10 0 18.5-7.3 19.9-16.8.1-.8.1-1.6.1-2.7z" />
    </svg>
  );
}
function DropboxLogo() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#0061FF" d="M14 6l10 7-10 7-10-7L14 6zm20 0l10 7-10 7-10-7 10-7zM4 28l10-7 10 7-10 7-10-7zm40 0l-10-7-10 7 10 7 10-7z" />
    </svg>
  );
}
function BoxLogo() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#1F8DED" d="M8 12h32a4 4 0 014 4v16a4 4 0 01-4 4H8a4 4 0 01-4-4V16a4 4 0 014-4z" />
      <path fill="#fff" d="M24 33c-4.4 0-8-3.6-8-8h4a4 4 0 108 0h4c0 4.4-3.6 8-8 8z" />
    </svg>
  );
}
function MegaLogo() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 48 48" aria-hidden="true">
      <circle cx="24" cy="24" r="20" fill="#E11D48" />
      <path d="M14 31V17h4l6 6 6-6h4v14h-4V23l-6 6-6-6v8h-4z" fill="#fff" />
    </svg>
  );
}
function TeamsLogo() {
  // Simple Teams-like glyph (no external assets)
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 48 48" aria-hidden="true">
      <defs>
        <linearGradient id="tg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#6264A7" />
          <stop offset="1" stopColor="#464775" />
        </linearGradient>
      </defs>
      <rect x="6" y="10" width="28" height="28" rx="6" fill="url(#tg)" />
      <rect x="30" y="16" width="12" height="16" rx="3" fill="#8B8CC7" />
      <path fill="#fff" d="M20 18h-6v3h3v9h3v-9h3v-3h-3z" />
    </svg>
  );
}
function SlackLogo() {
  // Minimal Slack-like glyph
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 48 48" aria-hidden="true">
      <rect x="8" y="20" width="8" height="8" rx="4" fill="#36C5F0" />
      <rect x="20" y="8" width="8" height="8" rx="4" fill="#2EB67D" />
      <rect x="32" y="20" width="8" height="8" rx="4" fill="#E01E5A" />
      <rect x="20" y="32" width="8" height="8" rx="4" fill="#ECB22E" />
    </svg>
  );
}
