import React, { useEffect, useMemo, useRef, useState } from "react";

const icons = {
  new: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
    </svg>
  ),
  chat: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M4 4h16v12H5.17L4 17.17V4zm0-2a2 2 0 00-2 2v20l4-4h14a2 2 0 002-2V4a2 2 0 00-2-2H4z"/>
    </svg>
  ),
  search: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M15.5 14h-.79l-.28-.27A6.5 6.5 0 109.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79L20 21.5 21.5 20 15.5 14zM9.5 14a4.5 4.5 0 110-9 4.5 4.5 0 010 9z"/>
    </svg>
  ),
  close: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M18.3 5.71L12 12m0 0L5.7 5.7M12 12l6.3 6.3M12 12L5.7 18.3" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  ),
};

function formatWhen(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  const t = new Date();
  return d.toDateString() === t.toDateString()
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString();
}

export default function SearchChatsModal({
  isOpen,
  onClose,
  onSelectChat,
  onNewChat,
}) {
  const [query, setQuery] = useState("");
  const [chats, setChats] = useState([]);
  const [highlight, setHighlight] = useState({ section: "special", index: 0 });
  const inputRef = useRef(null);

  // load chats on open
  useEffect(() => {
    if (!isOpen) return;
    (async () => {
      try {
        const r = await fetch("/api/chats", { credentials: "include", cache: "no-store" });
        const d = await r.json();
        const arr = Array.isArray(d) ? d : d?.chats || [];
        setChats(arr);
      } catch {
        setChats([]);
      }
    })();
  }, [isOpen]);

  // autofocus and close on Esc
  useEffect(() => {
    if (!isOpen) return;
    const t = setTimeout(() => inputRef.current?.focus(), 0);
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      if (e.key === "ArrowUp")   { e.preventDefault(); move(-1); }
      if (e.key === "Enter")     { e.preventDefault(); selectCurrent(); }
    };
    window.addEventListener("keydown", onKey);
    return () => { clearTimeout(t); window.removeEventListener("keydown", onKey); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, highlight, chats, query]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = q
      ? chats.filter(c => String(c.title || c.name || c.id).toLowerCase().includes(q))
      : chats;

    const now = Date.now();
    const d7 = 7 * 24 * 60 * 60 * 1000;
    const d30 = 30 * 24 * 60 * 60 * 1000;

    const sec = { last7: [], last30: [], older: [] };
    for (const c of list) {
      const when = new Date(c.updated_at || c.created_at || 0).getTime();
      const age = now - when;
      if (age <= d7) sec.last7.push(c);
      else if (age <= d30) sec.last30.push(c);
      else sec.older.push(c);
    }
    return sec;
  }, [chats, query]);

  const flatOrder = useMemo(() => {
    const out = [];
    out.push({ section: "special", index: 0 }); // New chat
    for (const s of ["last7","last30","older"]) {
      const arr = filtered[s];
      for (let i=0;i<arr.length;i++) out.push({ section: s, index: i });
    }
    return out;
  }, [filtered]);

  function move(delta) {
    const idx = flatOrder.findIndex(
      (p) => p.section === highlight.section && p.index === highlight.index
    );
    const next = Math.max(0, Math.min(flatOrder.length - 1, idx + delta));
    setHighlight(flatOrder[next]);
  }

  function selectCurrent() {
    const { section, index } = highlight;
    if (section === "special") {
      onClose();
      onNewChat?.();
      return;
    }
    const item =
      section === "last7" ? filtered.last7[index] :
      section === "last30" ? filtered.last30[index] :
      filtered.older[index];

    if (item) {
      onClose();
      const id = item.id || item.chat_id || item._id;
      onSelectChat?.(id);
    }
  }

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center p-4">
      {/* backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      {/* modal */}
      <div className="relative mt-16 w-full max-w-2xl rounded-2xl bg-[#111827] shadow-2xl border border-white/10 overflow-hidden">
        {/* search bar */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10">
          <span className="text-white/70">{icons.search}</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search chats..."
            className="flex-1 bg-transparent text-white outline-none placeholder-white/50"
          />
          <button onClick={onClose} className="text-white/60 hover:text-white">{icons.close}</button>
        </div>

        {/* results */}
        <div className="max-h-[60vh] overflow-y-auto p-1">
          {/* New chat */}
          <ResultRow
            active={highlight.section === "special"}
            icon={icons.new}
            title="New chat"
            onClick={() => { onClose(); onNewChat?.(); }}
          />

          <Section title="Previous 7 Days" show={filtered.last7.length > 0} />
          {filtered.last7.map((c, i) => (
            <ResultRow
              key={(c.id || c._id || c.chat_id) + "-7"}
              active={highlight.section === "last7" && highlight.index === i}
              icon={icons.chat}
              title={c.title || c.name || `Chat ${(c.id || c._id || c.chat_id)}`}
              meta={formatWhen(c.updated_at || c.created_at)}
              onClick={() => { onClose(); onSelectChat?.(c.id || c._id || c.chat_id); }}
            />
          ))}

          <Section title="Previous 30 Days" show={filtered.last30.length > 0} />
          {filtered.last30.map((c, i) => (
            <ResultRow
              key={(c.id || c._id || c.chat_id) + "-30"}
              active={highlight.section === "last30" && highlight.index === i}
              icon={icons.chat}
              title={c.title || c.name || `Chat ${(c.id || c._id || c.chat_id)}`}
              meta={formatWhen(c.updated_at || c.created_at)}
              onClick={() => { onClose(); onSelectChat?.(c.id || c._id || c.chat_id); }}
            />
          ))}

          <Section title="Older" show={filtered.older.length > 0} />
          {filtered.older.map((c, i) => (
            <ResultRow
              key={(c.id || c._id || c.chat_id) + "-o"}
              active={highlight.section === "older" && highlight.index === i}
              icon={icons.chat}
              title={c.title || c.name || `Chat ${(c.id || c._id || c.chat_id)}`}
              meta={formatWhen(c.updated_at || c.created_at)}
              onClick={() => { onClose(); onSelectChat?.(c.id || c._id || c.chat_id); }}
            />
          ))}

          {filtered.last7.length + filtered.last30.length + filtered.older.length === 0 && (
            <div className="px-4 py-6 text-sm text-white/60">No chats found.</div>
          )}
        </div>
      </div>
    </div>
  );
}

function Section({ title, show }) {
  if (!show) return null;
  return (
    <div className="px-4 pt-3 pb-1 text-[11px] uppercase tracking-wide text-white/40">{title}</div>
  );
}

function ResultRow({ active, icon, title, meta, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-3 px-4 py-2 rounded-lg text-left transition
        ${active ? "bg-white/10 text-white" : "text-white/90 hover:bg-white/5"}`}
    >
      <span className="text-white/80">{icon}</span>
      <div className="flex-1 min-w-0">
        <div className="truncate">{title}</div>
        {meta && <div className="truncate text-xs text-white/60">{meta}</div>}
      </div>
    </button>
  );
}