// src/components/ChatInput.jsx — input with FULL-WIDTH inline quote block (inside the composer)
import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  FiMic, FiMicOff, FiSend, FiPlus, FiChevronDown,
  FiMail, FiCalendar, FiPaperclip, FiX
} from "react-icons/fi";
import { TbBrandGoogleDrive, TbBrandOnedrive, TbCloud } from "react-icons/tb";
import { SiDropbox, SiBox } from "react-icons/si";
import { FaMicrosoft } from "react-icons/fa";
import { PiMicrosoftTeamsLogoFill } from "react-icons/pi";

const ACCEPT = [
  ".pdf",".txt",".md",".doc",".docx",".ppt",".pptx",".xls",".xlsx",".csv",
  ".png",".jpg",".jpeg",".gif",".tiff",".bmp",".heic",".rtf",".yml",".yaml",".json"
].join(",");

// ChatPanel listens to this to close the file picker before a normal send
export const RESUME_EVENT = "file-select:ensure-resumed";

export default function ChatInput({ onSend, disabled, userInput, setUserInput }) {
  const BRAND = "#E0C389";
  const FRAME = "rgba(255,255,255,0.08)";

  /* ===== Voice ===== */
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);
  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    const rec = new SR();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onstart = () => setIsListening(true);
    rec.onend = () => setIsListening(false);
    rec.onerror = () => setIsListening(false);
    rec.onresult = (e) => {
      const t = e.results?.[0]?.[0]?.transcript || "";
      if (t) { setUserInput(t); setTimeout(() => doSend(t), 120); }
    };
    recognitionRef.current = rec;
  }, [setUserInput]);
  const toggleVoice = () => {
    const r = recognitionRef.current;
    if (!r) return;
    try { isListening ? r.stop() : r.start(); } catch {}
  };

  /* ===== AskGPT + quote listeners ===== */
  const taRef = useRef(null);
  const [quote, setQuote] = useState(""); // quoted selection shown INSIDE the input

  useEffect(() => {
    const onInsert = (e) => {
      const t = e?.detail?.text || "";
      if (!t) return;
      setUserInput(t);
      requestAnimationFrame(() => taRef.current?.focus());
    };
    const onSendText = (e) => {
      const t = e?.detail?.text || "";
      if (!t.trim()) return;
      window.dispatchEvent(new Event(RESUME_EVENT));
      setUserInput(t);
      doSend(t);
    };
    const onSetQuote = (e) => {
      const t = (e?.detail?.text || "").trim();
      if (!t) return;
      setQuote(t);
      requestAnimationFrame(() => taRef.current?.focus());
    };

    window.addEventListener("chat:insert-text", onInsert);
    window.addEventListener("chat:send-text", onSendText);
    window.addEventListener("chat:set-quote", onSetQuote);
    return () => {
      window.removeEventListener("chat:insert-text", onInsert);
      window.removeEventListener("chat:send-text", onSendText);
      window.removeEventListener("chat:set-quote", onSetQuote);
    };
  }, [setUserInput]);

  /* ===== textarea ===== */
  const [focused, setFocused] = useState(false);
  useEffect(() => { if (!disabled) requestAnimationFrame(() => taRef.current?.focus()); }, [disabled]);
  useEffect(() => { autosize(); }, [userInput]);
  const autosize = () => {
    const ta = taRef.current; if (!ta) return;
    ta.style.height = "0px"; ta.style.height = Math.min(180, ta.scrollHeight) + "px";
  };

  /* ===== Connections (unchanged) ===== */
  const [connections, setConnections] = useState({
    sharepoint: { connected: false, account_email: null },
    onedrive:   { connected: false, account_email: null },
    google:     { connected: false, account_email: null },
    dropbox:    { connected: false, account_email: null },
    box:        { connected: false, account_email: null },
    mega:       { connected: false, account_email: null },
    outlook:           { connected: false, account_email: null },
    gmail:             { connected: false, account_email: null },
    outlook_calendar:  { connected: false, account_email: null },
    google_calendar:   { connected: false, account_email: null },
    teams:             { connected: false, account_email: null },
  });

  const canonicalKey = (raw) => {
    const k = (raw || "").toLowerCase().replace(/[\s_.-]+/g, "");
    const map = {
      sharepoint: ["sharepoint","mssharepoint"],
      onedrive: ["onedrive","msonedrive"],
      google: ["googledrive","google","gdrive","google_drive","googledrivefiles"],
      dropbox: ["dropbox"],
      box: ["box"],
      mega: ["mega","meganz"],
      outlook: ["outlook","msoutlook","o365mail"],
      outlook_calendar: ["outlookcalendar","msoutlookcalendar","outlookcal","o365calendar"],
      gmail: ["gmail"],
      google_calendar: ["googlecalendar","gcalendar","googlecal","google_calendar"],
      teams: ["teams","msteams","microsoftteams","ms_teams","ms-teams"],
    };
    for (const [key, aliases] of Object.entries(map)) if (aliases.includes(k)) return key;
    if (k.includes("sharepoint")) return "sharepoint";
    if (k.includes("onedrive")) return "onedrive";
    if (k.includes("outlook") && k.includes("calendar")) return "outlook_calendar";
    if (k.includes("google") && k.includes("calendar")) return "google_calendar";
    if (k.includes("outlook")) return "outlook";
    if (k.includes("gmail")) return "gmail";
    if (k.includes("google")) return "google";
    if (k.includes("dropbox")) return "dropbox";
    if (k.includes("mega")) return "mega";
    if (k.includes("box")) return "box";
    if (k.includes("teams")) return "teams";
    return null;
  };

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/api/connections", {
          credentials: "include", cache: "no-store", headers: { "Cache-Control": "no-cache" },
        });
        const data = await r.json().catch(() => ({}));
        const list = Array.isArray(data?.connections) ? data.connections : [];
        const map = {};
        for (const it of list) {
          const key = canonicalKey(it.provider || it.type || it.name);
          const connected = !!(it.connected ?? it.is_connected ?? it.active ?? (it.status === "connected") ?? it.status === true);
          if (key) map[key] = { connected, account_email: it.account_email || null };
        }
        const msConnected = !!(map.sharepoint?.connected || map.onedrive?.connected || map.outlook?.connected || map.outlook_calendar?.connected || map.teams?.connected);
        const gConnected  = !!(map.google?.connected || map.gmail?.connected || map.google_calendar?.connected);
        if (msConnected) {
          const msEmail = map.sharepoint?.account_email || map.onedrive?.account_email || map.outlook?.account_email || map.outlook_calendar?.account_email || map.teams?.account_email || null;
          if (!map.outlook)          map.outlook          = { connected: !!msConnected, account_email: msEmail };
          if (!map.outlook_calendar) map.outlook_calendar = { connected: !!msConnected, account_email: msEmail };
          if (!map.teams)            map.teams            = { connected: !!msConnected, account_email: msEmail };
        }
        if (gConnected) {
          const gEmail = map.google?.account_email || map.gmail?.account_email || map.google_calendar?.account_email || null;
          if (!map.gmail)           map.gmail           = { connected: true, account_email: gEmail };
          if (!map.google_calendar) map.google_calendar = { connected: true, account_email: gEmail };
        }
        setConnections((prev) => ({ ...prev, ...map }));
      } catch {}
    })();
  }, []);

  /* ===== Source flags (unchanged) ===== */
  const flagOrNull = (k) => {
    const v = sessionStorage.getItem(k);
    return v === null ? null : v === "1";
  };
  const [useSharePoint, setUseSharePoint] = useState(flagOrNull("src_sp")   ?? true);
  const [useOneDrive,   setUseOneDrive]   = useState(flagOrNull("src_od")   ?? true);
  const [useGoogle,     setUseGoogle]     = useState(flagOrNull("src_gd")   ?? true);
  const [useDropbox,    setUseDropbox]    = useState(flagOrNull("src_dbx")  ?? true);
  const [useBox,        setUseBox]        = useState(flagOrNull("src_box")  ?? true);
  const [useMega,       setUseMega]       = useState(flagOrNull("src_mega") ?? true);
  const [useOutlook,    setUseOutlook]    = useState(flagOrNull("src_ol")   ?? true);
  const [useGmail,      setUseGmail]      = useState(flagOrNull("src_gm")   ?? true);
  const [useOutlookCal, setUseOutlookCal] = useState(flagOrNull("src_ocal") ?? true);
  const [useGoogleCal,  setUseGoogleCal]  = useState(flagOrNull("src_gcal") ?? true);
  const [useTeams,      setUseTeams]      = useState(flagOrNull("src_teams")?? true);

  useEffect(() => {
    const keys = ["src_sp","src_od","src_gd","src_dbx","src_box","src_mega","src_ol","src_gm","src_ocal","src_gcal","src_teams"];
    const noFlags = keys.every((k) => sessionStorage.getItem(k) === null);
    if (!noFlags) return;
    const defaults = {
      sp:    !!connections.sharepoint.connected,
      od:    !!connections.onedrive.connected,
      gd:    !!connections.google.connected,
      dbx:   !!connections.dropbox.connected,
      box:   !!connections.box.connected,
      mega:  !!connections.mega.connected,
      ol:    !!connections.outlook.connected || (!!connections.sharepoint.connected || !!connections.onedrive.connected),
      gm:    !!connections.gmail.connected   || !!connections.google.connected,
      ocal:  !!connections.outlook_calendar.connected || (!!connections.sharepoint.connected || !!connections.onedrive.connected),
      gcal:  !!connections.google_calendar.connected  || !!connections.google.connected,
      teams: !!connections.teams.connected || (!!connections.sharepoint.connected || !!connections.onedrive.connected),
    };
    setUseSharePoint(defaults.sp);  setUseOneDrive(defaults.od);  setUseGoogle(defaults.gd);
    setUseDropbox(defaults.dbx);    setUseBox(defaults.box);      setUseMega(defaults.mega);
    setUseOutlook(defaults.ol);     setUseGmail(defaults.gm);     setUseOutlookCal(defaults.ocal);
    setUseGoogleCal(defaults.gcal); setUseTeams(defaults.teams);
    Object.entries({
      src_sp:defaults.sp,src_od:defaults.od,src_gd:defaults.gd,src_dbx:defaults.dbx,src_box:defaults.box,
      src_mega:defaults.mega,src_ol:defaults.ol,src_gm:defaults.gm,src_ocal:defaults.ocal,src_gcal:defaults.gcal,src_teams:defaults.teams
    }).forEach(([k,v])=>sessionStorage.setItem(k, v ? "1" : "0"));
    window.dispatchEvent(new CustomEvent("sources:sync", { detail: defaults }));
  }, [connections]);

  useEffect(() => { sessionStorage.setItem("src_sp",    useSharePoint ? "1" : "0"); }, [useSharePoint]);
  useEffect(() => { sessionStorage.setItem("src_od",    useOneDrive   ? "1" : "0"); }, [useOneDrive]);
  useEffect(() => { sessionStorage.setItem("src_gd",    useGoogle     ? "1" : "0"); }, [useGoogle]);
  useEffect(() => { sessionStorage.setItem("src_dbx",   useDropbox    ? "1" : "0"); }, [useDropbox]);
  useEffect(() => { sessionStorage.setItem("src_box",   useBox        ? "1" : "0"); }, [useBox]);
  useEffect(() => { sessionStorage.setItem("src_mega",  useMega       ? "1" : "0"); }, [useMega]);
  useEffect(() => { sessionStorage.setItem("src_ol",    useOutlook    ? "1" : "0"); }, [useOutlook]);
  useEffect(() => { sessionStorage.setItem("src_gm",    useGmail      ? "1" : "0"); }, [useGmail]);
  useEffect(() => { sessionStorage.setItem("src_ocal",  useOutlookCal ? "1" : "0"); }, [useOutlookCal]);
  useEffect(() => { sessionStorage.setItem("src_gcal",  useGoogleCal  ? "1" : "0"); }, [useGoogleCal]);
  useEffect(() => { sessionStorage.setItem("src_teams", useTeams      ? "1" : "0"); }, [useTeams]);

  useEffect(() => {
    const handler = (e) => {
      const s = e.detail || {};
      if ("sp"    in s) setUseSharePoint(!!s.sp);
      if ("od"    in s) setUseOneDrive(!!s.od);
      if ("gd"    in s) setUseGoogle(!!s.gd);
      if ("dbx"   in s) setUseDropbox(!!s.dbx);
      if ("box"   in s) setUseBox(!!s.box);
      if ("mega"  in s) setUseMega(!!s.mega);
      if ("ol"    in s) setUseOutlook(!!s.ol);
      if ("gm"    in s) setUseGmail(!!s.gm);
      if ("ocal"  in s) setUseOutlookCal(!!s.ocal);
      if ("gcal"  in s) setUseGoogleCal(!!s.gcal);
      if ("teams" in s) setUseTeams(!!s.teams);
    };
    window.addEventListener("sources:sync", handler);
    return () => window.removeEventListener("sources:sync", handler);
  }, []);

  /* ===== Sources dropdown ===== */
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const sourcesRef = useRef(null);
  useEffect(() => {
    const click = (e) => sourcesRef.current && !sourcesRef.current.contains(e.target) && setSourcesOpen(false);
    const esc = (e) => e.key === "Escape" && setSourcesOpen(false);
    document.addEventListener("mousedown", click);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", click); document.removeEventListener("keydown", esc); };
  }, []);
  const activeSourcesArray = useCallback(() => {
    const a = [];
    if (connections.sharepoint.connected && useSharePoint) a.push("sharepoint");
    if (connections.onedrive.connected   && useOneDrive)   a.push("onedrive");
    if (connections.google.connected     && useGoogle)     a.push("google");
    if (connections.dropbox.connected    && useDropbox)    a.push("dropbox");
    if (connections.box.connected        && useBox)        a.push("box");
    if (connections.mega.connected       && useMega)       a.push("mega");
    if (connections.outlook.connected          && useOutlook)    a.push("outlook");
    if (connections.gmail.connected            && useGmail)      a.push("gmail");
    if (connections.outlook_calendar.connected && useOutlookCal) a.push("outlook_calendar");
    if (connections.google_calendar.connected  && useGoogleCal)  a.push("google_calendar");
    if (connections.teams.connected            && useTeams)      a.push("teams");
    return a;
  }, [connections,useSharePoint,useOneDrive,useGoogle,useDropbox,useBox,useMega,useOutlook,useGmail,useOutlookCal,useGoogleCal,useTeams]);

  /* ===== Uploads ===== */
  const [uploads, setUploads] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  const addFiles = async (fileList) => {
    const files = Array.from(fileList || []).filter(f => !!f && !!f.name);
    if (!files.length) return;
    setIsUploading(true);
    try {
      const fd = new FormData();
      for (const f of files) fd.append("files", f);
      const r = await fetch("/api/uploads/files", { method: "POST", credentials: "include", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.error || "Upload failed");
      const saved = (Array.isArray(data.files) ? data.files : []).map(f => ({
        ...f,
        id: f.id || f.path || (Date.now() + "-" + Math.random().toString(36).slice(2)),
      }));
      setUploads(prev => [...prev, ...saved]);
      window.dispatchEvent(new CustomEvent("uploads:added", { detail: saved }));
    } catch (e) {
      alert(e.message || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  const removeUpload = async (id) => {
    setUploads(prev => {
      const hit = prev.find(u => u.id === id);
      const next = prev.filter(u => u.id !== id);
      if (hit?.path) {
        fetch("/api/uploads/files", {
          method: "DELETE",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: hit.path }),
        }).catch(() => {});
      }
      return next;
    });
  };

  const onPickClick = () => fileInputRef.current?.click();
  const onFilePicked = (e) => addFiles(e.target.files);
  const onDrop = (e) => { e.preventDefault(); e.stopPropagation(); setDragging(false); if (!disabled) addFiles(e.dataTransfer?.files || []); };
  const onDragOver = (e) => { e.preventDefault(); e.stopPropagation(); if (!dragging) setDragging(true); };
  const onDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setDragging(false); };

  /* ===== Send ===== */
  const doSend = (textOverride) => {
    const instruction = (textOverride ?? userInput).trim();
    const hasQuote = !!quote.trim();
    const message = hasQuote ? `${instruction}\n\n"""${quote.trim()}"""` : instruction;
    if (!message && uploads.length === 0) return;

    window.dispatchEvent(new Event(RESUME_EVENT));

    const attachments = uploads.map(u => ({ ...u }));
    onSend(message, null, activeSourcesArray(), { attachments });

    setUserInput("");
    setQuote("");
    setUploads([]);
    requestAnimationFrame(() => taRef.current?.focus());
    setSourcesOpen(false);
  };
  const onKeyDown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doSend(); } };

  return (
    <div className="px-4 pb-5 pt-3" onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}>
      <div className="mx-auto max-w-[980px]">

        {/* Attachment chips (outside composer, unchanged) */}
        {uploads.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {uploads.map((u) => (
              <div key={u.id} className="group flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-2 py-1 text-[12px] text-zinc-200">
                <span className="inline-flex items-center gap-1">
                  <span className="inline-block h-5 w-5 rounded-lg bg-white/10 grid place-items-center">
                    <FiPaperclip size={12} />
                  </span>
                  <span className="max-w-[220px] truncate">{u.name || "file"}</span>
                </span>
                <button onClick={() => removeUpload(u.id)} className="opacity-70 hover:opacity-100" title="Remove">
                  <FiX size={14} />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Composer wrapper with ring */}
        <div className="relative group">
          <div
            className={`pointer-events-none absolute -inset-0.5 rounded-[28px] blur transition-opacity duration-300 ${focused ? "opacity-100" : "opacity-60"}`}
            style={{ background: "conic-gradient(from 180deg at 50% 50%, rgba(224,195,137,.65), rgba(124,58,237,.35), rgba(56,189,248,.35), rgba(224,195,137,.65))",
                     boxShadow: focused ? "0 0 40px rgba(224,195,137,.25)" : "none" }}
          />
          <div className="relative rounded-[26px] p-[2px]" style={{ background: FRAME }}>
            {/* Composer body as a COLUMN: full-width quote on top, then the row */}
            <div className="relative rounded-[24px] border border-white/10 bg-[#0f152b]/70 px-3 py-2 backdrop-blur-xl shadow-[0_10px_30px_rgba(0,0,0,.5)] flex flex-col gap-2">

              {/* FULL-WIDTH quote block */}
              {quote && (
                <div className="w-full rounded-xl border border-white/10 bg-white/[.04] px-3 py-2 text-[13px] text-zinc-200">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <div className="font-medium text-zinc-100">Quoted text</div>
                    <button
                      onClick={() => setQuote("")}
                      className="text-zinc-300 hover:text-white"
                      title="Remove quote"
                    >
                      <FiX size={16} />
                    </button>
                  </div>
                  <div className="rounded-lg bg-black/30 border border-white/10 px-3 py-2 whitespace-pre-wrap">
                    {quote}
                  </div>
                </div>
              )}

              {/* Row: left icons + textarea + right icons */}
              <div className="flex items-end gap-2">
                {/* Left: connectors & sources */}
                <button
                  type="button"
                  onClick={() => (window.location.href = "/dashboard")}
                  className="group rounded-full p-2 text-zinc-300 hover:text-white hover:bg-white/5"
                  title="Connectors"
                  disabled={disabled}
                >
                  <FiPlus size={18} />
                </button>

                <SourcesDropdown
                  open={sourcesOpen}
                  setOpen={setSourcesOpen}
                  refEl={sourcesRef}
                  connections={connections}
                  useSharePoint={useSharePoint} setUseSharePoint={setUseSharePoint}
                  useOneDrive={useOneDrive}     setUseOneDrive={setUseOneDrive}
                  useGoogle={useGoogle}         setUseGoogle={setUseGoogle}
                  useDropbox={useDropbox}       setUseDropbox={setUseDropbox}
                  useBox={useBox}               setUseBox={setUseBox}
                  useMega={useMega}             setUseMega={setUseMega}
                  useOutlook={useOutlook}       setUseOutlook={setUseOutlook}
                  useGmail={useGmail}           setUseGmail={setUseGmail}
                  useOutlookCal={useOutlookCal} setUseOutlookCal={setUseOutlookCal}
                  useGoogleCal={useGoogleCal}   setUseGoogleCal={setUseGoogleCal}
                  useTeams={useTeams}           setUseTeams={setUseTeams}
                  activeCount={activeSourcesArray().length}
                />

                {/* Textarea expands */}
                <textarea
                  ref={taRef}
                  autoFocus
                  rows={1}
                  className="flex-1 resize-none bg-transparent px-2 pb-1 pt-1 text-[15px] text-white placeholder:zinc-400 outline-none"
                  placeholder={isUploading ? "Uploading…" : "Type your prompt here…"}
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyDown={onKeyDown}
                  onFocus={() => setFocused(true)}
                  onBlur={() => setFocused(false)}
                  disabled={disabled}
                />

                {/* Right controls */}
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept={ACCEPT}
                  onChange={onFilePicked}
                  className="hidden"
                />
                <button
                  type="button"
                  onClick={onPickClick}
                  className="relative grid place-items-center rounded-full p-2 text-zinc-300 hover:text-white hover:bg-white/5 disabled:opacity-50"
                  title="Add photos & files"
                  disabled={disabled}
                >
                  <FiPaperclip size={18} />
                </button>
                <button
                  type="button"
                  onClick={toggleVoice}
                  className="relative grid place-items-center rounded-full p-2 text-zinc-300 hover:text-white hover:bg-white/5 disabled:opacity-50"
                  title={isListening ? "Stop voice input" : "Voice input"}
                  disabled={disabled}
                >
                  {isListening && <span className="absolute inset-0 -m-1 animate-ping rounded-full" style={{ backgroundColor: BRAND, opacity: 0.25 }} />}
                  {isListening ? <FiMicOff size={18} /> : <FiMic size={18} />}
                </button>
                <button
                  type="button"
                  onClick={() => doSend()}
                  className="grid place-items-center rounded-full p-2 text-[#1b1720] shadow-lg disabled:opacity-50"
                  title="Send"
                  disabled={disabled || isUploading}
                  style={{ background: "linear-gradient(180deg, #F1DFB1 0%, #E0C389 50%, #CBA360 100%)" }}
                >
                  <FiSend size={18} />
                </button>
              </div>
            </div>

            {/* Drag overlay */}
            {dragging && (
              <div className="absolute inset-0 rounded-[24px] border-2 border-dashed border-white/20 bg-white/5 grid place-items-center pointer-events-none">
                <div className="rounded-xl bg-black/50 px-4 py-2 text-sm text-zinc-200">Drop files to upload</div>
              </div>
            )}
          </div>
        </div>

        <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-400">
          <span>Enter to send • Shift+Enter for newline</span>
          <span className="hidden sm:inline">
            {isUploading ? "Uploading…" : "Icon pills expand on hover/click"}
          </span>
        </div>
      </div>
    </div>
  );
}

/* ------------------------ helpers ------------------------ */

function SourcesDropdown(props) {
  const {
    open, setOpen, refEl, connections, activeCount,
    useSharePoint, setUseSharePoint,
    useOneDrive, setUseOneDrive,
    useGoogle, setUseGoogle,
    useDropbox, setUseDropbox,
    useBox, setUseBox,
    useMega, setUseMega,
    useOutlook, setUseOutlook,
    useGmail, setUseGmail,
    useOutlookCal, setUseOutlookCal,
    useGoogleCal, setUseGoogleCal,
    useTeams, setUseTeams,
  } = props;

  return (
    <div className="relative" ref={refEl}>
      <button
        type="button"
        onClick={() => setOpen((s) => !s)}
        className="relative flex items-center gap-1 rounded-full p-2 text-zinc-300 hover:text-white hover:bg-white/5"
        title="Choose Sources"
      >
        <span className="relative inline-flex -space-x-1">
          <i className="h-2.5 w-2.5 rounded-full bg-emerald-400 inline-block" />
          <i className="h-2.5 w-2.5 rounded-full bg-sky-400 inline-block" />
          <i className="h-2.5 w-2.5 rounded-full bg-blue-500 inline-block" />
        </span>
        <FiChevronDown size={16} className="opacity-70" />
        {activeCount > 0 && (
          <span className="ml-1 rounded-full bg-white/10 px-1.5 py-0.5 text-[10px] text-zinc-200">{activeCount}</span>
        )}
      </button>

      {open && (
        <div className="absolute left-0 bottom-11 w=[360px] md:w-[360px] w-[320px] rounded-2xl border border-white/10 bg-[#0b1124]/95 backdrop-blur-xl shadow-2xl">
          <div className="px-3 py-2 text-xs font-semibold text-zinc-300 border-b border-white/10">
            Search connected sources
          </div>
          <div className="p-2">
            {/* Files */}
            <RowToggle
              icon={<TbBrandGoogleDrive size={18} className="text-emerald-400" />}
              title="Google Drive"
              subtitle={connections.google.connected ? "Connected" : "Not connected"}
              checked={connections.google.connected && useGoogle}
              disabled={!connections.google.connected}
              onChange={setUseGoogle}
            />
            <RowToggle
              icon={<TbBrandOnedrive size={18} className="text-sky-400" />}
              title="OneDrive"
              subtitle={connections.onedrive.connected ? "Connected" : "Not connected"}
              checked={connections.onedrive.connected && useOneDrive}
              disabled={!connections.onedrive.connected}
              onChange={setUseOneDrive}
            />
            <RowToggle
              icon={<FaMicrosoft size={16} className="text-teal-400" />}
              title="SharePoint"
              subtitle={connections.sharepoint.connected ? "Connected" : "Not connected"}
              checked={connections.sharepoint.connected && useSharePoint}
              disabled={!connections.sharepoint.connected}
              onChange={setUseSharePoint}
            />
            <RowToggle
              icon={<SiDropbox size={18} className="text-sky-500" />}
              title="Dropbox"
              subtitle={connections.dropbox.connected ? "Connected" : "Not connected"}
              checked={connections.dropbox.connected && useDropbox}
              disabled={!connections.dropbox.connected}
              onChange={setUseDropbox}
            />
            <RowToggle
              icon={<SiBox size={18} className="text-blue-500" />}
              title="Box"
              subtitle={connections.box.connected ? "Connected" : "Not connected"}
              checked={connections.box.connected && useBox}
              disabled={!connections.box.connected}
              onChange={setUseBox}
            />
            <RowToggle
              icon={<TbCloud size={18} className="text-rose-400" />}
              title="MEGA"
              subtitle={connections.mega.connected ? "Connected" : "Not connected"}
              checked={connections.mega.connected && useMega}
              disabled={!connections.mega.connected}
              onChange={setUseMega}
            />

            {/* Mail/Calendars/Teams */}
            <div className="mt-3 mb-1 px-1 text-[11px] uppercase tracking-wide text-white/40">Mail • Calendars • Teams</div>

            <RowToggle
              icon={<FiMail size={16} className="text-sky-300" />}
              title="Outlook Mail"
              subtitle={connections.outlook.connected ? "Connected" : "Not connected"}
              checked={connections.outlook.connected && useOutlook}
              disabled={!connections.outlook.connected}
              onChange={setUseOutlook}
            />
            <RowToggle
              icon={<FiMail size={16} className="text-emerald-300" />}
              title="Gmail"
              subtitle={connections.gmail.connected ? "Connected" : "Not connected"}
              checked={connections.gmail.connected && useGmail}
              disabled={!connections.gmail.connected}
              onChange={setUseGmail}
            />
            <RowToggle
              icon={<FiCalendar size={16} className="text-sky-300" />}
              title="Outlook Calendar"
              subtitle={connections.outlook_calendar.connected ? "Connected" : "Not connected"}
              checked={connections.outlook_calendar.connected && useOutlookCal}
              disabled={!connections.outlook_calendar.connected}
              onChange={setUseOutlookCal}
            />
            <RowToggle
              icon={<FiCalendar size={16} className="text-emerald-300" />}
              title="Google Calendar"
              subtitle={connections.google_calendar.connected ? "Connected" : "Not connected"}
              checked={connections.google_calendar.connected && useGoogleCal}
              disabled={!connections.google_calendar.connected}
              onChange={setUseGoogleCal}
            />
            <RowToggle
              icon={<PiMicrosoftTeamsLogoFill size={17} className="text-indigo-300" />}
              title="Microsoft Teams"
              subtitle={connections.teams.connected ? "Connected" : "Not connected"}
              checked={connections.teams.connected && useTeams}
              disabled={!connections.teams.connected}
              onChange={setUseTeams}
            />

            <div className="mt-2 rounded-lg bg-white/[.04] px-3 py-2 text-[11px] leading-snug text-zinc-300">
              Toggle which connected sources should be searched for this prompt.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function RowToggle({ icon, title, subtitle, checked, onChange, disabled }) {
  return (
    <div className="flex items-center justify-between rounded-xl px-2 py-2 hover:bg-white/[.04]">
      <div className="flex items-center gap-3">
        <span className="w-6 flex justify-center">{icon}</span>
        <div>
          <div className="text-sm text-white">{title}</div>
          <div className="text-[11px] text-zinc-400">{subtitle}</div>
        </div>
      </div>
      <label className={`relative inline-flex cursor-pointer items-center ${disabled ? "opacity-30 cursor-not-allowed" : ""}`}>
        <input type="checkbox" className="peer sr-only" checked={!!checked} onChange={(e) => !disabled && onChange(e.target.checked)} disabled={disabled} />
        <div className="peer h-5 w-9 rounded-full bg-white/20 after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:bg-white after:transition-all peer-checked:bg-emerald-500 peer-checked:after:translate-x-4" />
      </label>
    </div>
  );
}
