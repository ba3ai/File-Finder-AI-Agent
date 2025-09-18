// src/components/HeaderModeMenu.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";

/* tiny ui bits */
const Caret = ({ open }) => (
  <svg width="14" height="14" viewBox="0 0 24 24" className={`transition ${open ? "rotate-180" : ""}`} fill="currentColor">
    <path d="M7 10l5 5 5-5" />
  </svg>
);
const Dot = ({ on }) => (
  <span className={`inline-block h-2 w-2 rounded-full ${on ? "bg-emerald-400" : "bg-white/40"}`} />
);

/* event bus */
const SCOPE_EVENT = "app:scope:set";

/* Keys */
const FILE_KEYS = ["sharepoint", "onedrive", "google", "dropbox", "box", "mega"];
// Communications: mail, calendars, and Teams
const MC_KEYS   = ["outlook", "gmail", "outlook_calendar", "google_calendar", "teams"];
const ALL_KEYS  = [...FILE_KEYS, ...MC_KEYS];

const LABELS = {
  all: "Auto",
  sharepoint: "SharePoint",
  onedrive: "OneDrive",
  google: "Google Drive",
  dropbox: "Dropbox",
  box: "Box",
  mega: "MEGA",
  outlook: "Outlook Mail",
  gmail: "Gmail",
  outlook_calendar: "Outlook Calendar",
  google_calendar: "Google Calendar",
  teams: "Microsoft Teams",
};

/* normalize connections array/object into a map with virtual providers derived */
function normalizeConnections(d) {
  const map = {};
  const arr = Array.isArray(d) ? d : Array.isArray(d?.connections) ? d.connections : null;

  if (arr) {
    for (const item of arr) {
      const raw = String(item.provider || item.type || item.name || "").toLowerCase();
      const ok =
        item.connected ?? item.is_connected ?? item.active ??
        (item.status === "connected" || item.status === true);
      if (!raw) continue;

      map[raw] = !!ok;

      // common alias -> canonical
      if (raw === "google_drive") map.google = map.google || !!ok;
    }

    // derive "microsoft" umbrella if either file connector is present
    if ("onedrive" in map || "sharepoint" in map) {
      map.microsoft = map.microsoft || map.onedrive || map.sharepoint;
    }
  } else if (d && typeof d === "object") {
    for (const k of Object.keys(d)) map[k.toLowerCase()] = !!d[k];
  }

  // Canonicalize Google Drive
  if (map.google_drive && !map.google) map.google = map.google_drive;

  // Derive virtual mail/calendar providers when the backend doesn’t list them explicitly
  if (map.microsoft) {
    if (!("outlook" in map)) map.outlook = true;
    if (!("outlook_calendar" in map)) map.outlook_calendar = true;
    // NOTE: Do NOT auto-enable Teams; it requires separate consent.
  }
  if (map.google) {
    if (!("gmail" in map)) map.gmail = true;
    if (!("google_calendar" in map)) map.google_calendar = true;
  }

  return map;
}

const isConnected = (key, map) => {
  switch (key) {
    // Files
    case "sharepoint": return !!(map.sharepoint || map.microsoft);
    case "onedrive":   return !!(map.onedrive   || map.microsoft);
    case "google":     return !!map.google;
    case "dropbox":    return !!map.dropbox;
    case "box":        return !!map.box;
    case "mega":       return !!map.mega;
    // Mail & Calendars & Teams
    case "outlook":           return !!(map.outlook || map.microsoft);
    case "outlook_calendar":  return !!(map.outlook_calendar || map.outlook || map.microsoft);
    case "gmail":             return !!(map.gmail || map.google);
    case "google_calendar":   return !!(map.google_calendar || map.google);
    case "teams":             return !!map.teams; // separate connector
    default: return true;
  }
};

/** compute & broadcast flags (to ChatInput & others) */
function pushSync(scope, connMap) {
  const flags = {
    // files
    sp:false, od:false, gd:false, dbx:false, box:false, mega:false,
    // communications
    ol:false, gm:false, ocal:false, gcal:false, teams:false,
  };

  if (scope === "all") {    // Auto = all connected (files + mail + calendars + Teams)
    for (const k of FILE_KEYS) {
      const on = isConnected(k, connMap);
      if (k === "sharepoint") flags.sp   = on;
      if (k === "onedrive")   flags.od   = on;
      if (k === "google")     flags.gd   = on;
      if (k === "dropbox")    flags.dbx  = on;
      if (k === "box")        flags.box  = on;
      if (k === "mega")       flags.mega = on;
    }
    for (const k of MC_KEYS) {
      const on = isConnected(k, connMap);
      if (k === "outlook")           flags.ol    = on;
      if (k === "gmail")             flags.gm    = on;
      if (k === "outlook_calendar")  flags.ocal  = on;
      if (k === "google_calendar")   flags.gcal  = on;
      if (k === "teams")             flags.teams = on;
    }
  } else {
    // Single-source selection
    flags.sp   = scope === "sharepoint"       && isConnected("sharepoint", connMap);
    flags.od   = scope === "onedrive"         && isConnected("onedrive",   connMap);
    flags.gd   = scope === "google"           && isConnected("google",     connMap);
    flags.dbx  = scope === "dropbox"          && isConnected("dropbox",    connMap);
    flags.box  = scope === "box"              && isConnected("box",        connMap);
    flags.mega = scope === "mega"             && isConnected("mega",       connMap);

    flags.ol    = scope === "outlook"          && isConnected("outlook",           connMap);
    flags.gm    = scope === "gmail"            && isConnected("gmail",             connMap);
    flags.ocal  = scope === "outlook_calendar" && isConnected("outlook_calendar",  connMap);
    flags.gcal  = scope === "google_calendar"  && isConnected("google_calendar",   connMap);
    flags.teams = scope === "teams"            && isConnected("teams",             connMap);
  }

  // persist & broadcast
  sessionStorage.setItem("src_sp",    flags.sp    ? "1" : "0");
  sessionStorage.setItem("src_od",    flags.od    ? "1" : "0");
  sessionStorage.setItem("src_gd",    flags.gd    ? "1" : "0");
  sessionStorage.setItem("src_dbx",   flags.dbx   ? "1" : "0");
  sessionStorage.setItem("src_box",   flags.box   ? "1" : "0");
  sessionStorage.setItem("src_mega",  flags.mega  ? "1" : "0");

  sessionStorage.setItem("src_ol",    flags.ol    ? "1" : "0");
  sessionStorage.setItem("src_gm",    flags.gm    ? "1" : "0");
  sessionStorage.setItem("src_ocal",  flags.ocal  ? "1" : "0");
  sessionStorage.setItem("src_gcal",  flags.gcal  ? "1" : "0");
  sessionStorage.setItem("src_teams", flags.teams ? "1" : "0");

  window.dispatchEvent(new CustomEvent("sources:sync", { detail: flags }));
}

/**
 * Props:
 * - scope ('all' | connector key)
 * - setScope(scope)
 * - connections (array or map; normalized internally)
 * - onActiveSourcesChange(activeSourcesArray)
 */
export default function HeaderModeMenu({
  scope = "all",
  setScope = () => {},
  connections = {},
  onActiveSourcesChange = () => {},
}) {
  const connMap = useMemo(() => normalizeConnections(connections), [connections]);
  const hasAny = useMemo(() => ALL_KEYS.some((k) => isConnected(k, connMap)), [connMap]);

  // When scope or connection availability changes, tell App and ChatInput
  useEffect(() => {
    if (!hasAny) return; // wait until we actually know what’s connected
    if (scope === "all") {
      const active = [...FILE_KEYS, ...MC_KEYS].filter((k) => isConnected(k, connMap)); // Auto: all connected
      onActiveSourcesChange(active);
    } else {
      onActiveSourcesChange([scope]);
    }
    pushSync(scope, connMap); // will also persist flags into sessionStorage
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope, connMap, hasAny]);

  // Listen for sidebar scope changes
  useEffect(() => {
    const onScopeEvent = (e) => {
      const next = e?.detail?.scope;
      const src  = e?.detail?.source;
      if (!next || src === "header") return; // ignore our own events
      if (next !== scope) setScope(next);
    };
    window.addEventListener(SCOPE_EVENT, onScopeEvent);
    return () => window.removeEventListener(SCOPE_EVENT, onScopeEvent);
  }, [scope, setScope]);

  // UI
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const close = (e) => { if (!ref.current) return; if (!ref.current.contains(e.target)) setOpen(false); };
    const esc = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, []);

  const displayLabel = LABELS[scope] || "Auto";

  // Emit helper for header-originated changes
  const emitScope = (next) => window.dispatchEvent(new CustomEvent(SCOPE_EVENT, { detail: { scope: next, source: "header" } }));

  const choose = (next) => {
    setScope(next);
    emitScope(next);
    setOpen(false);
  };

  return (
    <div className="relative" ref={ref}>
      <button
        className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-sm text-white hover:bg-white/10"
        onClick={() => setOpen((o) => !o)}
        title="Choose source"
        type="button"
      >
        <span className="opacity-90">{displayLabel}</span>
        <Caret open={open} />
      </button>

      {open && (
        <div className="absolute left-0 mt-2 z-50 w-[260px] rounded-xl border border-white/10 bg-[#0b1324] shadow-2xl">
          <div className="p-2" role="menu" aria-label="Select source">
            {/* Auto */}
            <MenuItem
              title="Auto"
              active={scope === "all"}
              disabled={FILE_KEYS.every((k) => !isConnected(k, connMap)) && MC_KEYS.every((k) => !isConnected(k, connMap))}
              right={<Dot on />}
              onClick={() => choose("all")}
            />

            <div className="my-1 h-px bg-white/10" />

            {/* File sources */}
            {FILE_KEYS.map((k) => {
              const connected = isConnected(k, connMap);
              return (
                <MenuItem
                  key={k}
                  title={LABELS[k]}
                  active={scope === k}
                  disabled={!connected}
                  right={<Dot on={connected} />}
                  onClick={() => connected && choose(k)}
                />
              );
            })}

            <div className="my-2 h-px bg-white/10" />

            {/* Mail, Calendars & Teams */}
            {MC_KEYS.map((k) => {
              const connected = isConnected(k, connMap);
              return (
                <MenuItem
                  key={k}
                  title={LABELS[k]}
                  active={scope === k}
                  disabled={!connected}
                  right={<Dot on={connected} />}
                  onClick={() => connected && choose(k)}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function MenuItem({ title, right, onClick, active, disabled }) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={`w-full text-left rounded-lg px-3 py-2 flex items-center gap-2 ${
        disabled
          ? "opacity-50 cursor-not-allowed"
          : active
            ? "bg-white/10 ring-1 ring-white/15"
            : "hover:bg-white/5"
      }`}
      role="menuitemradio"
      aria-checked={!!active}
    >
      <div className="min-w-0">
        <div className="text-sm text-white truncate">{title}</div>
      </div>
      <div className="ml-auto flex items-center gap-2">{right}</div>
    </button>
  );
}
