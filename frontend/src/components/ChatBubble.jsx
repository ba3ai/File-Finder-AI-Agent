// src/components/ChatBubble.jsx
import React from "react";
import { LuBot } from "react-icons/lu";
import { FiUser, FiPaperclip } from "react-icons/fi";

/** Animated status chip (thinking/searching/access/error). */
function StatusChip({ type = "thinking" }) {
  const map = {
    thinking:          { label: "Thinking",        cls: "bg-[#1a2a52] border-[#25407a]" },
    "searching-file":  { label: "Searching files", cls: "bg-[#1a2a52] border-[#25407a]" },
    "checking-access": { label: "Checking access", cls: "bg-[#1a2a52] border-[#25407a]" },
    error:             { label: "Error",           cls: "bg-[#3b1f1f] border-[#6d2b2b]" },
  };
  const cfg = map[type] || map.thinking;
  const animated = type !== "error";
  return (
    <span
      role="status"
      aria-live="polite"
      className={`inline-flex items-center text-xs px-2.5 py-1 rounded-full border mr-2 mb-1 ${cfg.cls} ${
        animated ? "animate-pulse" : ""
      }`}
    >
      <span>{cfg.label}</span>
      {animated && (
        <span className="ml-1 inline-flex items-end gap-[2px]" aria-hidden="true">
          <span className="h-1.5 w-1.5 rounded-full bg-white/90 animate-bounce [animation-duration:1.1s]" />
          <span className="h-1.5 w-1.5 rounded-full bg-white/70 animate-bounce [animation-delay:.15s] [animation-duration:1.1s]" />
          <span className="h-1.5 w-1.5 rounded-full bg-white/50 animate-bounce [animation-delay:.3s]  [animation-duration:1.1s]" />
        </span>
      )}
    </span>
  );
}

/** Turn markdown links and naked URLs into <a> (no innerHTML). */
function renderWithLinks(text) {
  const nodes = [];
  const str = String(text ?? "");
  const MD  = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
  const URL = /(https?:\/\/[^\s<>()]+[^\s<>().,;!?"')\]])/g;
  let i = 0;
  while (i < str.length) {
    MD.lastIndex = i; URL.lastIndex = i;
    const m1 = MD.exec(str); const m2 = URL.exec(str);
    let m = null, type = null;
    if (m1 && m2) m = (m1.index <= m2.index ? (type="md", m1) : (type="url", m2));
    else if (m1)  m = (type="md", m1);
    else if (m2)  m = (type="url", m2);
    if (!m) { nodes.push(str.slice(i)); break; }
    if (m.index > i) nodes.push(str.slice(i, m.index));
    if (type === "md") {
      const label = m[1], href = m[2];
      nodes.push(
        <a
          key={`md-${m.index}`}
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="underline text-blue-300 hover:text-blue-200 break-words"
        >
          {label}
        </a>
      );
      i = MD.lastIndex;
    } else {
      const href = m[1];
      nodes.push(
        <a
          key={`url-${m.index}`}
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="underline text-blue-300 hover:text-blue-200 break-words"
        >
          {href}
        </a>
      );
      i = URL.lastIndex;
    }
  }
  return nodes;
}

export default function ChatBubble({
  sender = "AI",
  message = "",
  attachments = [],           // [{ id, name, url, path, serverId?, mime?, thumbUrl? }]
  isStatus = false,
  statusType = "thinking",
  timestamp,
  userAvatarUrl,
  aiAvatarUrl,
}) {
  const isUser =
    String(sender).toLowerCase() === "you" ||
    String(sender).toLowerCase() === "user";

  const [userImgFailed, setUserImgFailed] = React.useState(false);
  const [aiImgFailed, setAiImgFailed] = React.useState(false);

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"} py-1`}>
      <div className="relative max-w-[80%]">
        {/* Bubble */}
        <div
          className={[
            "rounded-2xl border shadow-md px-4 py-3 whitespace-pre-wrap break-words",
            isUser
              ? "bg-[#1a2a52] border-[#25407a] text-white"
              : "bg-[#0f1a33] border-[#1f2b4a] text-white",
          ].join(" ")}
        >
          {isStatus && <StatusChip type={statusType} />}

          <div className="text-[1rem]">{renderWithLinks(message)}</div>

          {/* Attachments */}
          {Array.isArray(attachments) && attachments.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2" aria-label="Message attachments">
              {attachments.map((a) => {
                const url =
                  a.url || a.previewUrl ||
                  (a.serverId ? `/uploads/chat/${a.serverId}` : undefined);
                const key =
                  a.id || a.serverId || a.path || a.name ||
                  Math.random().toString(36).slice(2);
                const name = a.name || a.filename || "file";
                const mime = String(a.mime || "").toLowerCase();
                const isImg =
                  mime.startsWith("image/") ||
                  /\.(png|jpe?g|gif|bmp|webp|tiff|heic)$/i.test(name);

                return (
                  <div
                    key={key}
                    className="rounded-lg border border-white/10 bg-white/5 p-2"
                    title={name}
                  >
                    {isImg && url ? (
                      <a href={url} target="_blank" rel="noopener noreferrer" className="block">
                        <img
                          src={a.thumbUrl || url}
                          alt={name}
                          className="max-w-[240px] rounded-md"
                          loading="lazy"
                        />
                        <div className="mt-1 text-[11px] text-blue-200 max-w-[240px] truncate">
                          {name}
                        </div>
                      </a>
                    ) : (
                      <a
                        href={url || "#"}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-[12px] text-blue-200 hover:text-white"
                      >
                        <span className="inline-block h-4 w-4 rounded bg-white/10 grid place-items-center">
                          <FiPaperclip size={12} />
                        </span>
                        <span className="max-w-[220px] truncate">{name}</span>
                      </a>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {!!timestamp && (
            <div className="mt-1 text-[11px] text-white/50">
              {new Date(timestamp).toLocaleString()}
            </div>
          )}
        </div>

        {/* gradient connector */}
        <div
          className={[
            "pointer-events-none absolute top-3 h-[2px] rounded-full opacity-80",
            isUser
              ? "-right-6 w-6 bg-gradient-to-l from-[#E0C389] to-transparent"
              : "-left-6  w-6 bg-gradient-to-r from-[#E0C389] to-transparent",
          ].join(" ")}
        />

        {/* Icon / Avatar badge */}
        <div
          className={[
            "absolute top-1 w-7 h-7 rounded-full overflow-hidden grid place-items-center shadow",
            "border backdrop-blur-sm",
            isUser
              ? "-right-8 bg-[#2a3a6b] border-[#3b4f8a] text-white"
              : "-left-8  bg-[#1f2b4a] border-[#2a3b7a] text-white",
          ].join(" ")}
          aria-hidden="true"
          title={isUser ? "You" : "AI"}
        >
          {isUser ? (
            userAvatarUrl && !userImgFailed ? (
              <img
                src={userAvatarUrl}
                alt="Your avatar"
                className="w-full h-full object-cover"
                onError={() => setUserImgFailed(true)}
                loading="lazy"
              />
            ) : (
              <FiUser size={20} />
            )
          ) : aiAvatarUrl && !aiImgFailed ? (
            <img
              src={aiAvatarUrl}
              alt="AI"
              className="w-full h-full object-cover"
              onError={() => setAiImgFailed(true)}
              loading="lazy"
            />
          ) : (
            <LuBot size={20} />
          )}
        </div>
      </div>
    </div>
  );
}
