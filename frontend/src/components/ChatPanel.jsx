// src/components/ChatPanel.jsx
import React, { useEffect, useRef, useState } from "react";
import ChatBubble from "./ChatBubble";

// ChatInput fires this right before it sends a normal message
const RESUME_EVENT = "file-select:ensure-resumed";

/** Optional helper for /api/chats (used by the quick switcher) */
async function apiGet(url) {
  const r = await fetch(url, {
    credentials: "include",
    cache: "no-store",
    headers: { "Cache-Control": "no-cache" },
  });
  if (r.status === 402 && window.location.pathname !== "/trial-ended") {
    window.location.replace("/trial-ended");
    throw new Error("trial_expired");
  }
  return r;
}

function ChatSwitcher({ onSelectChat, onNewChat }) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [chats, setChats] = useState([]);

  const load = async () => {
    setLoading(true);
    try {
      const res = await apiGet("/api/chats");
      const data = await res.json();
      const list = Array.isArray(data) ? data : data?.chats || [];
      setChats(list);
    } catch {}
    setLoading(false);
  };

  useEffect(() => { if (open) load(); }, [open]);

  return (
    <>
      <button
        type="button"
        title="Chats"
        className="fixed bottom-24 right-6 z-30 w-12 h-12 rounded-full bg-[#E0C389] text-[#1b1720] shadow-xl grid place-items-center hover:brightness-95"
        onClick={() => setOpen(true)}
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M4 4h16v12H7l-3 3V4z"/></svg>
      </button>

      {open && (
        <>
          <div className="fixed inset-0 bg-black/40 z-30" onClick={() => setOpen(false)} />
          <div className="fixed bottom-6 right-6 z-40 w-[380px] max-h-[70vh] overflow-y-auto custom-scrollbar rounded-2xl border border-[#1f2b4a] bg-[#0f1a33] text-white shadow-2xl">
            <div className="p-4 flex items-center justify-between border-b border-[#1f2b4a]">
              <h3 className="font-semibold">Chats</h3>
              <button onClick={() => setOpen(false)} className="text-white/70 hover:text-white">✕</button>
            </div>
            <div className="p-4">
              <button
                onClick={() => { setOpen(false); onNewChat && onNewChat(); }}
                className="w-full rounded-xl bg-[#BD945B] text-black px-3 py-2 font-medium hover:bg-[#a17d4b]"
              >
                + New chat
              </button>

              <div className="mt-3 space-y-2">
                {loading && <div className="text-sm text-white/70">Loading…</div>}
                {!loading && chats.length === 0 && <div className="text-sm text-white/60">No chats yet.</div>}

                {chats.map((c) => {
                  const id = c.id || c.chat_id || c._id;
                  const title = c.title || c.name || `Chat ${id}`;
                  const when = c.updated_at || c.created_at || c.ts || c.time;
                  return (
                    <button
                      key={id}
                      onClick={() => { setOpen(false); onSelectChat && onSelectChat(id); }}
                      className="w-full text-left rounded-xl bg-[#182544] border border-[#22345e] hover:bg-[#1d2a51] p-3"
                    >
                      <div className="flex items-center justify-between">
                        <div className="font-medium">{title}</div>
                        {when && <div className="text-xs text-white/60">{new Date(when).toLocaleString()}</div>}
                      </div>
                      {c.last && <div className="mt-1 text-xs text-white/60 line-clamp-1">{c.last}</div>}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}

export default function ChatPanel({
  messages,
  fileOptions,
  allFiles,
  pauseGPT,                 // true when the file picker should be shown
  toggleSelectFile,
  selectedFiles = [],
  sendSelectedFiles,
  page,
  totalFiles,
  onPageChange,
  onFilterChange,
  onSkipFileSelection,      // collapses the picker
  availableFileTypes,
  onSelectChat,
  onNewChat,
  userAvatarUrl,
  summarizeSelectedFiles,   // (selectedFiles) => Promise<void>  – parent handles POST + assistant bubble
  onUserMessage,            // (text, attachments?) => void     – parent appends user bubble locally
}) {
  const [busySummarize, setBusySummarize] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, fileOptions, pauseGPT]);

  // Close the picker when ChatInput sends a normal message
  useEffect(() => {
    const handler = () => { if (pauseGPT && typeof onSkipFileSelection === "function") onSkipFileSelection(); };
    window.addEventListener(RESUME_EVENT, handler);
    return () => window.removeEventListener(RESUME_EVENT, handler);
  }, [pauseGPT, onSkipFileSelection]);

  const getFileTypes = () => availableFileTypes || [];
  const isChecked = (id) => selectedFiles?.some((f) => f.id === id);

  const handleSummarize = async () => {
    if (busySummarize) return;
    if (!selectedFiles || selectedFiles.length === 0) return;

    setBusySummarize(true);

    // 1) Append a user bubble locally (no server fetch here)
    const text = "Summarize this file";
    if (typeof onUserMessage === "function") {
      onUserMessage(text);
    } else {
      // Fallback, only if your app uses events to append
      window.dispatchEvent(new CustomEvent("chat:append-user", { detail: { text } }));
    }

    // 2) Run summarization
    try {
      if (typeof summarizeSelectedFiles === "function") {
        await summarizeSelectedFiles(selectedFiles); // Parent will append the AI message
      } else {
        // Fallback direct call (optional)
        const body = { selectedIds: (selectedFiles || []).map((f) => f.id), prompt: text };
        const res = await fetch("/api/summarize_selected", {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        const ai = data?.response || "⚠️ Summarization failed.";
        window.dispatchEvent(new CustomEvent("chat:append-assistant", { detail: { text: ai } }));
      }
    } catch (e) {
      console.error("Summarize failed:", e);
      window.dispatchEvent(new CustomEvent("chat:append-assistant", { detail: { text: "⚠️ Summarization failed." } }));
    } finally {
      // 3) Collapse the picker and resume input
      if (typeof onSkipFileSelection === "function") onSkipFileSelection();
      window.dispatchEvent(new Event(RESUME_EVENT));
      setBusySummarize(false);
    }
  };

  return (
    <div className="w-full max-w-[980px] mx-auto p-4 space-y-4">
      <ChatSwitcher onSelectChat={onSelectChat} onNewChat={onNewChat} />

      {messages?.map((msg, i) => (
        <ChatBubble
          key={i}
          sender={msg.sender}
          message={msg.message}
          isStatus={msg.isStatus}
          statusType={msg.statusType}
          timestamp={msg.timestamp}
          userAvatarUrl={userAvatarUrl}
          sources={msg.sources}
          // IMPORTANT: render from `attachments`
          attachments={msg.attachments || msg.files || []}
        />
      ))}

      <div ref={chatEndRef} />

      {/* Selection stage UI */}
      {pauseGPT && (
        <div className="bg-[#171717] p-6 rounded-2xl shadow-md border border-[#22345e] space-y-4 w-full">
          <h2 className="text-white text-xl font-semibold mb-2">📂 Files I Found for You</h2>

          {/* Filter */}
          <div className="flex justify-end mb-2">
            <label className="text-sm text-white mr-2">Filter:</label>
            <select
              className="text-sm bg-[#182544] text-white px-2 py-1 rounded border border-[#22345e]"
              onChange={(e) => onFilterChange(e.target.value)}
              defaultValue=""
            >
              <option value="">All</option>
              {getFileTypes().map((ext) => (
                <option key={ext} value={ext}>
                  {ext.toUpperCase().replace(".", "")}
                </option>
              ))}
            </select>
          </div>

          {/* Files */}
          {fileOptions?.length > 0 ? (
            <div className="space-y-3">
              {fileOptions.map((file, index) => (
                <div
                  key={file.id}
                  className="bg-[#182544] rounded-lg p-4 hover:bg-[#1d2a51] transition-colors cursor-pointer border border-[#22345e]"
                  onClick={() => toggleSelectFile(file)}
                >
                  <div className="flex justify-between items-center mb-2">
                    <h3 className="text-white font-medium text-base">
                      {(page - 1) * 5 + index + 1}. {file.name}
                    </h3>
                    <input
                      type="checkbox"
                      className="form-checkbox text-[#BD945B] h-5 w-5 pointer-events-none"
                      checked={isChecked(file.id)}
                      readOnly
                    />
                  </div>

                  <div className="flex justify-between items-center text-sm text-blue-300">
                    {file.webUrl ? (
                      <a
                        href={file.webUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:underline"
                        onClick={(e) => e.stopPropagation()}
                      >
                        🔗 Preview File
                      </a>
                    ) : (
                      <span className="opacity-70">No preview</span>
                    )}
                    <span className="text-white/70">File #{(page - 1) * 5 + index + 1}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-white text-sm italic">⚠️ No files found for this filter.</p>
          )}

          {/* Pagination */}
          <div className="flex justify-between items-center mt-4">
            <button
              className={`text-sm font-medium py-2 px-4 rounded-lg transition ${
                page === 1
                  ? "bg-[#1d2a51] text-white/50 cursor-not-allowed"
                  : "bg-[#BD945B] text-black hover:bg-[#a17d4b]"
              }`}
              onClick={() => onPageChange(page - 1)}
              disabled={page === 1}
            >
              ← Previous
            </button>

            <span className="text-white/80 text-sm">
              Page {page} of {Math.max(1, Math.ceil((totalFiles || 0) / 5))}
            </span>

            <button
              className={`text-sm font-medium py-2 px-4 rounded-lg transition ${
                page >= Math.max(1, Math.ceil((totalFiles || 0) / 5))
                  ? "bg-[#1d2a51] text-white/50 cursor-not-allowed"
                  : "bg-[#BD945B] text-black hover:bg-[#a17d4b]"
              }`}
              onClick={() => onPageChange(page + 1)}
              disabled={page >= Math.max(1, Math.ceil((totalFiles || 0) / 5))}
            >
              Next →
            </button>
          </div>

          {/* Send files */}
          <button
            className={`w-full mt-4 py-2 px-4 text-white font-semibold rounded-lg shadow-md transition ${
              selectedFiles.length > 0
                ? "bg-[#BD945B] hover:bg-[#a17d4b] text-black"
                : "bg-[#1d2a51] cursor-not-allowed"
            }`}
            onClick={sendSelectedFiles}
            disabled={selectedFiles.length === 0}
          >
            Send Selected Files
          </button>

          {/* Summarize files – follows chat flow */}
          <button
            className={`w-full mt-2 py-2 px-4 text-white font-semibold rounded-lg shadow-md transition ${
              selectedFiles.length > 0 && !busySummarize
                ? "bg-[#25406d] hover:bg-[#2c4a7c]"
                : "bg-[#1d2a51] cursor-not-allowed"
            }`}
            onClick={handleSummarize}
            disabled={selectedFiles.length === 0 || busySummarize}
            title={busySummarize ? "Summarizing…" : "Summarize selected files"}
          >
            {busySummarize ? "Summarizing…" : "Summarize Selected Files"}
          </button>

          <p className="text-xs text-white/60 text-center mt-2">
            Tip: You can keep chatting below without selecting any files.
          </p>
        </div>
      )}
    </div>
  );
}
