// src/components/Sidebar.jsx
import React, { useEffect, useRef, useState } from "react";
import SearchChatsModal from "./SearchChatsModal.jsx";

/* ---------- event bus ---------- */
const SCOPE_EVENT = "app:scope:set";

/* ---------- helpers ---------- */
const noop = () => {};
const shallowEqArr = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const shallowEqObj = (a, b) => {
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => a[k] === b[k]);
};
function formatWhen(ts) {
  if (!ts) return "";
  const d = new Date(ts), t = new Date();
  return d.toDateString() === t.toDateString()
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString();
}

/* ---------- icons (inline, React-safe) ---------- */
const icons = {
  compose: (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" className="icon" aria-hidden="true">
      <path d="M2.67 11.333V8.667c0-.923 0-1.655.049-2.244.049-.597.15-1.106.387-1.572l.154-.275A3.33 3.33 0 0 1 5.03 3.02c.418-.179.872-.263 1.395-.305C7.013 2.668 7.745 2.667 8.667 2.667h.5c.367 0 .665.298.665.666s-.298.665-.665.665h-.5c-.944 0-1.612 0-2.135.042-.386.032-.659.085-.876.162l-.2.086a2 2 0 0 0-.864.982l-.103.184c-.126.247-.206.561-.248 1.075-.043.522-.043 1.191-.043 2.135v2.666c0 .944 0 1.613.042 2.135.042.515.122.83.248 1.077l.103.184c.257.418.624.758 1.064.982l.2.086c.217.077.49.13.876.162.523.042 1.191.043 2.135.043h2.667c.944 0 1.612 0 2.135-.043.514-.042.829-.122 1.076-.248l.184-.103c.418-.257.759-.624.982-1.064l.086-.2c.077-.217.13-.49.162-.876.043-.522.043-1.191.043-2.135v-.5c0-.367.298-.665.666-.665.367 0 .665.298.665.665v.5c0 .923 0 1.655-.048 2.244-.043.522-.126.976-.306 1.394l-.083.177a3.33 3.33 0 0 1-1.472 1.593l-.275.154c-.466.237-.975.339-1.572.387-.589.049-1.322.049-2.244.049H8.667c-.922 0-1.655 0-2.244-.049a5.35 5.35 0 0 1-1.394-.306l-.177-.083A3.33 3.33 0 0 1 3.26 15.424l-.154-.275c-.237-.466-.339-.975-.387-1.572-.049-.589-.049-1.322-.049-2.244ZM13.465 3.113c.956-.78 2.365-.724 3.255.166l.167.184c.728.892.728 2.179 0 3.07l-.167.185-5.047 5.048a3 3 0 0 1-1.944.73l-.317.058-1.818.259a.667.667 0 0 1-.564-.188.667.667 0 0 1-.17-.564l.259-1.817.058-.318a3 3 0 0 1 .9-1.943l5.047-5.048.185-.167c.33-.404.33-.988 0-1.392l-.085-.082ZM15.78 4.221c-.404-.404-1.043-.43-1.476-.076l-.084.076-5.047 5.048c-.357.357-.605.807-.716 1.297l-.039.212-.135.938.939-.134.212-.04a2 2 0 0 0 1.298-.716l5.047-5.048.076-.084c.33-.404.33-.988 0-1.392l-.077-.084Z" />
    </svg>
  ),
  search: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M15.5 14h-.79l-.28-.27A6.5 6.5 0 1 0 9.5 16a6.47 6.47 0 0 0 4.23-1.57l.27.28v.79L20 21.5 21.5 20 15.5 14zm-6 0a4.5 4.5 0 1 1 0-9 4.5 4.5 0 0 1 0 9z"/></svg>
  ),
  auto: (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" className="icon" aria-hidden="true">
      <path d="M7.916 11c1.52 0 2.789 1.072 3.095 2.5h5.655c.302.062.53.33.53.651s-.228.589-.53.651H11.01c-.305 1.429-1.575 2.5-3.095 2.5-1.52-.002-2.79-1.073-3.095-2.5H3.333a.667.667 0 1 1 0-1.334h1.488c.305-1.428 1.575-2.499 3.095-2.499Zm0 1.33c-1.013.002-1.835.824-1.835 1.667 0 1.014.822 1.835 1.835 1.835 1.014 0 1.836-.821 1.836-1.835 0-1.013-.822-1.835-1.836-1.835Zm4.167-8.333c1.52 0 2.79 1.072 3.095 2.5h1.488a.667.667 0 0 1 0 1.334H15.18c-.305 1.428-1.575 2.5-3.095 2.5-1.52 0-2.79-1.072-3.095-2.5H3.333a.667.667 0 1 1 0-1.334h4.655c.305-1.428 1.575-2.5 3.095-2.5Zm0 1.33c-1.013 0-1.835.822-1.835 1.666 0 1.014.822 1.836 1.835 1.836 1.014 0 1.836-.822 1.836-1.836 0-.844-.822-1.666-1.836-1.666Z" />
    </svg>
  ),
  user: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10zm0 2c-4 0-8 2-8 6v2h16v-2c0-4-4-6-8-6z"/></svg>
  ),
  check: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.2 5.5 12.7 4 14.2 9 19l11-11-1.5-1.5z"/></svg>
  ),
  msCloud: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M7 18h10a4 4 0 0 0 0-8h-.2A6 6 0 0 0 5 9.5 3.5 3.5 0 0 0 7 18z"/></svg>
  ),
  gDrive: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4h5l5 8-5 8H7l5-8-5-8z"/></svg>
  ),
  dropbox: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M7 3l5 3-5 3-5-3 5-3zm10 0l5 3-5 3-5-3 5-3zM2 13l5 3 5-3-5-3-5 3zm20 0l-5 3-5-3 5-3 5 3zM7 17l5 3 5-3-5-3-5 3z"/></svg>
  ),
  box: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M3 7l9-5 9 5-9 5-9-5zm0 4l9 5 9-5v7l-9 5-9-5v-7z"/></svg>
  ),
  mega: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="10"/><path d="M7 15V9h2l3 3 3-3h2v6h-2v-3.5l-3 3-3-3V15H7z" fill="#0b1324"/></svg>
  ),
  mail: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M4 6h16a2 2 0 0 1 2 2v.2l-10 5.6L2 8.2V8a2 2 0 0 1 2-2zm0 4.4 8 4.5 8-4.5V16a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-5.6z"/></svg>
  ),
  calendar: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M7 2h2v2h6V2h2v2h3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h3V2zm13 8H4v10h16V10z"/></svg>
  ),
  teams: (
    <svg width="18" height="18" viewBox="0 0 48 48" fill="currentColor" aria-hidden="true">
      <path d="M20 14a6 6 0 1 1 0 12h-2v10c0 2.2 1.8 4 4 4h14a6 6 0 0 0 6-6V20h-6a6 6 0 0 1-6-6v-6H22a2 2 0 0 0-2 2v4h0Z" opacity=".2"/>
      <rect x="10" y="14" width="14" height="20" rx="3" />
      <circle cx="36" cy="12" r="6" />
      <path d="M14 18h10v4h-3v12h-4V22h-3v-4Z" fill="#0b1324"/>
    </svg>
  ),
  sparkle: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 2l2.5 6.5L21 11l-6.5 2.5L12 20l-2.5-6.5L3 11l6.5-2.5L12 2zM5 20l1.5-3.5L10 15l-3.5-1.5L5 10l-1.5 3.5L0 15l3.5 1.5L5 20zM24 9l-1-2.5L20.5 5 19 2l-1.5 3L14 5l3.5 1.5L19 10l1.5-3L24 9z"/>
    </svg>
  ),
  // NEW: workspace glyph (map pin)
  workspace: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 2a7 7 0 0 0-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 0 0-7-7zm0 9.5a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5z"/>
    </svg>
  ),
};

const Dot = ({ on }) => (
  <span className={`inline-block h-2 w-2 rounded-full ${on ? "bg-emerald-400" : "bg-white/40"}`} />
);

/* ---------- Sidebar component ---------- */
export default function Sidebar({
  onNewChat = noop,
  onSelectChat = noop,
  activeChatId,
  refreshFlag,
  onOpenProfile = noop,
  onLogout = noop,
  onScopeChange = noop,         // 'all' | connector key (single)
  onActiveSourcesChange = noop, // array form of the selection
  selectedScope: externalScope,
}) {
  const [me, setMe] = useState(null);
  const [chats, setChats] = useState([]);

  // Connected providers from API
  const [connMap, setConnMap] = useState({});

  // Workspace info (for label only)
  const [wsName, setWsName] = useState("");

  // UI
  const [acctOpen, setAcctOpen] = useState(false);
  const acctRef = useRef(null);
  const [searchOpen, setSearchOpen] = useState(false);

  // local scope when not controlled
  const [internalScope, setInternalScope] = useState("all");
  const scope = externalScope || internalScope;

  // routes
  const goToPricing = () => { window.location.href = "/pricing"; };
  const goToWorkspace = () => { window.location.href = "/workspace"; };

  /* ----- data load ----- */
  useEffect(() => { (async () => {
    try { const r = await fetch("/api/profile", { credentials: "include", cache: "no-store" }); setMe(await r.json()); } catch {}
  })(); }, []);
  useEffect(() => { (async () => {
    try { const r = await fetch("/api/chats", { credentials: "include", cache: "no-store" }); const d = await r.json(); setChats(Array.isArray(d) ? d : d?.chats || []); } catch { setChats([]); }
  })(); }, [refreshFlag]);
  useEffect(() => { (async () => {
    try {
      const r = await fetch("/api/connections", { credentials: "include", cache: "no-store" });
      const d = await r.json();
      setConnMap(normalizeConnections(d));
    } catch { setConnMap({}); }
  })(); }, []);
  // Load workspace name (best-effort; if 402, we simply hide the name)
  useEffect(() => { (async () => {
    try {
      const r = await fetch("/api/workspaces/me", { credentials: "include", cache: "no-store" });
      if (r.ok) {
        const j = await r.json();
        setWsName(j?.workspace?.name || "");
      } else {
        setWsName(""); // not on Team plan or no workspace yet
      }
    } catch { setWsName(""); }
  })(); }, []);

  /* account dropdown outside-click — only while open */
  useEffect(() => {
    if (!acctOpen) return;
    const onDocClick = (e) => { if (!acctRef.current) return; if (!acctRef.current.contains(e.target)) setAcctOpen(false); };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [acctOpen]);

  /* ----- connection normalization ----- */
  function canonicalize(name) {
    const raw = String(name || "").toLowerCase().trim();
    const compact = raw.replace(/[\s_.-]+/g, "");
    const map = {
      microsoft: "microsoft", ms: "microsoft",
      sharepoint: "sharepoint", mssharepoint: "sharepoint",
      onedrive: "onedrive", msonedrive: "onedrive",
      google: "google", googledrive: "google", gdrive: "google", googledrivefiles: "google", google_drive: "google",
      dropbox: "dropbox", box: "box", mega: "mega",
      outlook: "outlook", outlookcalendar: "outlook_calendar",
      gmail: "gmail", googlecalendar: "google_calendar",
      teams: "teams",
    };
    return map[compact] || raw;
  }

  function normalizeConnections(d) {
    const map = {};
    const arr = Array.isArray(d) ? d : Array.isArray(d?.connections) ? d.connections : null;
    if (arr) {
      for (const item of arr) {
        const prov = canonicalize(item.provider || item.type || item.name);
        const ok =
          item.connected ?? item.is_connected ?? item.active ??
          (item.status === "connected" || item.status === true);
        if (prov) map[prov] = !!ok;
      }
      if ("onedrive" in map || "sharepoint" in map) map.microsoft = map.microsoft || map.onedrive || map.sharepoint;
      if (map.microsoft) {
        if (!("outlook" in map)) map.outlook = true;
        if (!("outlook_calendar" in map)) map.outlook_calendar = true;
        if (!("teams" in map) && (map.sharepoint || map.onedrive)) map.teams = true;
      }
      if (map.google) {
        if (!("gmail" in map)) map.gmail = true;
        if (!("google_calendar" in map)) map.google_calendar = true;
      }
      return map;
    }
    if (d && typeof d === "object") {
      for (const k of Object.keys(d)) map[canonicalize(k)] = !!d[k];
    }
    return map;
  }

  const isConnected = (key, map = connMap) => {
    switch (key) {
      case "sharepoint":       return !!(map.sharepoint || map.microsoft);
      case "onedrive":         return !!(map.onedrive   || map.microsoft);
      case "google":           return !!map.google;
      case "dropbox":          return !!map.dropbox;
      case "box":              return !!map.box;
      case "mega":             return !!map.mega;

      case "outlook":          return !!(map.outlook || map.microsoft || map.sharepoint || map.onedrive);
      case "outlook_calendar": return !!(map.outlook_calendar || map.outlook || map.microsoft);
      case "gmail":            return !!(map.gmail || map.google);
      case "google_calendar":  return !!(map.google_calendar || map.google);

      case "teams":            return !!(map.teams || map.microsoft || map.sharepoint || map.onedrive);
      default:                 return true;
    }
  };

  /* ----- selection-only scope logic ----- */
  const allKeys = ["sharepoint", "onedrive", "google", "dropbox", "box", "mega"];

  const getActiveSources = (sc, cmap) => {
    if (sc !== "all") return [sc];
    return allKeys.filter((k) => isConnected(k, cmap));
  };
  const computeFlags = (sc, cmap) => {
    const flags = {
      sp: false, od: false, gd: false, dbx: false, box: false, mega: false,
      ol: false, ocal: false, gm: false, gcal: false,
      teams: false,
    };
    const setForKey = (k) => {
      const on = isConnected(k, cmap);
      if (k === "sharepoint") flags.sp = on;
      if (k === "onedrive")   flags.od = on;
      if (k === "google")     flags.gd = on;
      if (k === "dropbox")    flags.dbx = on;
      if (k === "box")        flags.box = on;
      if (k === "mega")       flags.mega = on;
      if (k === "outlook")          flags.ol = on;
      if (k === "outlook_calendar") flags.ocal = on;
      if (k === "gmail")            flags.gm = on;
      if (k === "google_calendar")  flags.gcal = on;
      if (k === "teams")            flags.teams = on;
    };

    if (sc === "all") {
      allKeys.forEach(setForKey);
    } else {
      setForKey(sc);
    }
    return flags;
  };

  const prevScopeRef  = useRef(scope);
  const prevActiveRef = useRef(getActiveSources(scope, connMap));
  const prevFlagsRef  = useRef(computeFlags(scope, connMap));

  useEffect(() => {
    const active = getActiveSources(scope, connMap);
    const flags = computeFlags(scope, connMap);

    const scopeChanged  = prevScopeRef.current !== scope;
    const activeChanged = !shallowEqArr(prevActiveRef.current, active);
    const flagsChanged  = !shallowEqObj(prevFlagsRef.current, flags);

    if (scopeChanged) onScopeChange(scope);
    if (activeChanged) onActiveSourcesChange(active);
    if (flagsChanged) {
      sessionStorage.setItem("src_sp",   flags.sp   ? "1" : "0");
      sessionStorage.setItem("src_od",   flags.od   ? "1" : "0");
      sessionStorage.setItem("src_gd",   flags.gd   ? "1" : "0");
      sessionStorage.setItem("src_dbx",  flags.dbx  ? "1" : "0");
      sessionStorage.setItem("src_box",  flags.box  ? "1" : "0");
      sessionStorage.setItem("src_mega", flags.mega ? "1" : "0");
      sessionStorage.setItem("src_ol",   flags.ol   ? "1" : "0");
      sessionStorage.setItem("src_ocal", flags.ocal ? "1" : "0");
      sessionStorage.setItem("src_gm",   flags.gm   ? "1" : "0");
      sessionStorage.setItem("src_gcal", flags.gcal ? "1" : "0");
      sessionStorage.setItem("src_teams", flags.teams ? "1" : "0");

      window.dispatchEvent(new CustomEvent("sources:sync", { detail: flags }));
    }

    prevScopeRef.current  = scope;
    prevActiveRef.current = active;
    prevFlagsRef.current  = flags;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope, connMap]);

  const emitScope = (nextScope) => {
    window.dispatchEvent(new CustomEvent(SCOPE_EVENT, { detail: { scope: nextScope, source: "sidebar" } }));
  };
  useEffect(() => {
    const onScopeEvent = (e) => {
      const next = e?.detail?.scope;
      const src  = e?.detail?.source;
      if (!next || src === "sidebar") return;
      if (next !== scope) {
        if (!externalScope) setInternalScope(next);
      }
    };
    window.addEventListener(SCOPE_EVENT, onScopeEvent);
    return () => window.removeEventListener(SCOPE_EVENT, onScopeEvent);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope, externalScope]);

  const setScopeLocal = (next) => {
    if (!externalScope) setInternalScope(next);
    emitScope(next);
  };

  /* ---------- UI rows ---------- */
  const Row = ({ icon, children, active, onClick, right, disabled }) => (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      onClick={() => { if (!disabled && typeof onClick === "function") onClick(); }}
      onKeyDown={(e) => { if (!disabled && (e.key === "Enter" || e.key === " ")) onClick?.(); }}
      className={`w-full flex items-center justify-between gap-3 rounded-lg px-3 py-2 text-sm transition
        ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
        ${active ? "bg-white/10 text-white ring-1 ring-white/20" : (!disabled ? "text-white/90 hover:bg-white/5" : "text-white/60")}
      `}
    >
      <span className="opacity-90">{icon}</span>
      <span className="truncate">{children}</span>
      <span className="ml-auto flex items-center gap-2">{right}</span>
    </div>
  );

  /* ---------- render ---------- */
  return (
    <aside id="sidebar-root" className="flex h-full w-[300px] flex-col bg-[#0b1324] text-white text-sm overflow-hidden">
      {/* Top actions */}
      <div className="px-2 pt-2 pb-2 border-b border-white/10">
        <Row icon={icons.compose} onClick={onNewChat}>New chat</Row>
        <Row icon={icons.search} onClick={() => setSearchOpen(true)}>Search chats</Row>

        {/* Connectors (selection only) */}
        <div className="mt-2 mb-1 text-[11px] uppercase tracking-wide text-white/40 px-3">Connectors</div>
        <div className="space-y-1">
          <Row
            icon={icons.auto}
            active={scope === "all"}
            onClick={() => setScopeLocal("all")}
            right={<Dot on />}
            disabled={["sharepoint","onedrive","google","dropbox","box","mega"].every((k) => !isConnected(k))}
          >
            Auto
          </Row>

          <Row icon={icons.msCloud} active={scope === "sharepoint"} onClick={() => setScopeLocal("sharepoint")} right={<Dot on={isConnected("sharepoint")} />} disabled={!isConnected("sharepoint")}>
            SharePoint
          </Row>

          <Row icon={icons.msCloud} active={scope === "onedrive"} onClick={() => setScopeLocal("onedrive")} right={<Dot on={isConnected("onedrive")} />} disabled={!isConnected("onedrive")}>
            OneDrive
          </Row>

          <Row icon={icons.gDrive} active={scope === "google"} onClick={() => setScopeLocal("google")} right={<Dot on={isConnected("google")} />} disabled={!isConnected("google")}>
            Google Drive
          </Row>

          <Row icon={icons.dropbox} active={scope === "dropbox"} onClick={() => setScopeLocal("dropbox")} right={<Dot on={isConnected("dropbox")} />} disabled={!isConnected("dropbox")}>
            Dropbox
          </Row>

          <Row icon={icons.box} active={scope === "box"} onClick={() => setScopeLocal("box")} right={<Dot on={isConnected("box")} />} disabled={!isConnected("box")}>
            Box
          </Row>

          <Row icon={icons.mega} active={scope === "mega"} onClick={() => setScopeLocal("mega")} right={<Dot on={isConnected("mega")} />} disabled={!isConnected("mega")}>
            MEGA
          </Row>

          {/* Collaboration */}
          <div className="mt-3 mb-1 text-[11px] uppercase tracking-wide text-white/40 px-3">Collaboration</div>
          <Row icon={icons.teams} active={scope === "teams"} onClick={() => setScopeLocal("teams")} right={<Dot on={isConnected("teams")} />} disabled={!isConnected("teams")}>
            Microsoft Teams
          </Row>

          {/* Mail & Calendar */}
          <div className="mt-3 mb-1 text-[11px] uppercase tracking-wide text-white/40 px-3">Mail & Calendars</div>

          <Row icon={icons.mail} active={scope === "outlook"} onClick={() => setScopeLocal("outlook")} right={<Dot on={isConnected("outlook")} />} disabled={!isConnected("outlook")}>
            Outlook Mail
          </Row>

          <Row icon={icons.mail} active={scope === "gmail"} onClick={() => setScopeLocal("gmail")} right={<Dot on={isConnected("gmail")} />} disabled={!isConnected("gmail")}>
            Gmail
          </Row>

          <Row icon={icons.calendar} active={scope === "outlook_calendar"} onClick={() => setScopeLocal("outlook_calendar")} right={<Dot on={isConnected("outlook_calendar")} />} disabled={!isConnected("outlook_calendar")}>
            Outlook Calendar
          </Row>

          <Row icon={icons.calendar} active={scope === "google_calendar"} onClick={() => setScopeLocal("google_calendar")} right={<Dot on={isConnected("google_calendar")} />} disabled={!isConnected("google_calendar")}>
            Google Calendar
          </Row>
        </div>
      </div>

      {/* section header */}
      <div className="px-2">
        <div className="text-[11px] uppercase tracking-wide text-white/40 px-3 pt-3">Chats</div>
      </div>

      {searchOpen && (
        <SearchChatsModal
          isOpen={searchOpen}
          onClose={() => setSearchOpen(false)}
          onSelectChat={onSelectChat}
          onNewChat={onNewChat}
        />
      )}

      {/* chat list */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-1">
        {chats.map((c) => {
          const id = c.id || c.chat_id || c._id;
          const title = c.title || c.name || `Chat ${id}`;
          const when = c.updated_at || c.created_at || c.ts || c.time;
          const active = id === activeChatId;
          return (
            <button
              key={id}
              type="button"
              onClick={() => onSelectChat(id)}
              className={`w-full rounded-lg px-3 py-2 text-left transition border ${
                active ? "bg-white text-black border-white/10" : "bg-white/5 text-white border-white/10 hover:bg-white/10"
              }`}
              aria-current={active ? "true" : "false"}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="truncate text-sm font-medium">{title}</div>
                <div className={`shrink-0 text-[10px] ${active ? "text-black/70" : "text-white/60"}`}>
                  {formatWhen(when)}
                </div>
              </div>
              {c.last && (
                <div className={`mt-0.5 truncate text-[11px] ${active ? "text-black/70" : "text-white/60"}`}>
                  {c.last}
                </div>
              )}
            </button>
          );
        })}
        {chats.length === 0 && <div className="px-2 py-3 text-xs text-white/60">No chats yet.</div>}
      </div>

      {/* account pill */}
      <div className="relative border-t border-white/10 p-3" ref={acctRef}>
        <button
          type="button"
          onClick={() => setAcctOpen((o) => !o)}
          className="flex w-full items-center gap-3 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-left hover:bg-white/10"
          title="Account"
        >
          <div className="grid h-8 w-8 place-items-center overflow-hidden rounded-full bg-white/10">
            {me?.avatar_url ? (
              <img src={me.avatar_url} alt="avatar" className="h-full w-full object-cover" />
            ) : (
              <span className="text-sm font-semibold text-white/80">
                {(me?.name || me?.email || "U").charAt(0).toUpperCase()}
              </span>
            )}
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm text-white">{me?.name || "Your profile"}</div>
            <div className="truncate text-[11px] text-white/60">{me?.email || "Personal account"}</div>
          </div>
          <div className="ml-auto opacity-60">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M7 10l5 5 5-5"/></svg>
          </div>
        </button>

        {acctOpen && (
          <div className="absolute bottom-14 left-3 right-3 z-50 rounded-xl border border-white/10 bg-[#111827] shadow-xl overflow-hidden">
            <div className="px-3 py-2 text-[11px] uppercase tracking-wide text-white/40">Account</div>

            {/* Upgrade plan */}
            <button
              type="button"
              onClick={() => { setAcctOpen(false); goToPricing(); }}
              className="w-full flex items-center gap-3 px-3 py-2 text-sm text-white/90 hover:bg-white/5"
            >
              <span className="grid place-items-center h-7 w-7 rounded-full bg-white/10">{icons.sparkle}</span>
              <div className="min-w-0 text-left">
                <div className="truncate">Upgrade plan</div>
                <div className="text-[11px] text-white/60 truncate">View pricing &amp; manage billing</div>
              </div>
            </button>

            {/* NEW: Workspace */}
            <button
              type="button"
              onClick={() => { setAcctOpen(false); goToWorkspace(); }}
              className="w-full flex items-center gap-3 px-3 py-2 text-sm text-white/90 hover:bg-white/5"
            >
              <span className="grid place-items-center h-7 w-7 rounded-full bg-white/10">{icons.workspace}</span>
              <div className="min-w-0 text-left">
                <div className="truncate">{wsName || "Workspace"}</div>
                <div className="text-[11px] text-white/60 truncate">{wsName ? "Manage members & seats" : "Team workspace"}</div>
              </div>
            </button>

            <button
              type="button"
              onClick={onOpenProfile}
              className="w-full flex items-center gap-3 px-3 py-2 text-sm text-white/90 hover:bg-white/5"
            >
              <span className="grid place-items-center h-7 w-7 rounded-full bg-white/10 overflow-hidden">
                {icons.user}
              </span>
              <div className="min-w-0 text-left">
                <div className="truncate">Profile &amp; Connections</div>
                <div className="text-[11px] text-white/60 truncate">{me?.email || ""}</div>
              </div>
            </button>

            <div className="mx-3 my-2 h-px bg-white/10" />
            <div className="px-3 pb-1 text-[11px] uppercase tracking-wide text-white/40">You</div>
            <div className="flex items-center gap-2 px-3 py-2 text-sm">
              <span className="grid place-items-center h-7 w-7 rounded-full bg-white/10">{icons.user}</span>
              <div className="min-w-0">
                <div className="text-white/90">Personal account</div>
                <div className="text-[11px] text-white/60">{me?.email || ""}</div>
              </div>
              <span className="ml-auto text-white/70">{icons.check}</span>
            </div>

            <div className="mx-3 my-2 h-px bg-white/10" />
            <button
              type="button"
              onClick={() => { setAcctOpen(false); onOpenProfile(); }}
              className="w-full text-left px-3 py-2 text-sm text-white/90 hover:bg-white/5"
            >
              Settings
            </button>
            <button type="button" className="w-full text-left px-3 py-2 text-sm text-white/90 hover:bg-white/5">
              Help
            </button>
            <div className="mx-3 my-2 h-px bg-white/10" />
            <button
              type="button"
              onClick={onLogout}
              className="w-full text-left px-3 py-2 text-sm text-white/90 hover:bg-white/5"
            >
              Log out
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}
