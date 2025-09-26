// src/components/AskGPT.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUturnLeftIcon } from "@heroicons/react/24/outline";

/**
 * AskGPT — shows 1–3 suggestion chips based on recent history.
 * Props:
 *  - history: [{ role: "user"|"assistant", content: string }]
 *  - onInsert: (text: string) => void
 *  - onSend?: (text: string) => void
 *  - autoFetch?: boolean (default true)
 *  - endpoint?: string (default "/askgpt/suggestions")
 *  - compact?: boolean (default false)
 */
export default function AskGPT({
  history = [],
  onInsert,
  onSend,
  autoFetch = true,
  endpoint = "/askgpt/suggestions",
  compact = false,
}) {
  const [loading, setLoading] = useState(false);
  const [sugs, setSugs] = useState([]);
  const abortRef = useRef(null);
  const lastReqKeyRef = useRef("");

  // Only send the last few turns to keep payloads tiny
  const body = useMemo(() => {
    const short = history.slice(-8);
    return JSON.stringify({ history: short, max: 3 });
  }, [history]);

  useEffect(() => {
    if (!autoFetch) return;

    // debounce to avoid rapid-fire requests while typing
    const handle = setTimeout(() => {
      const key = body;
      if (key === lastReqKeyRef.current) return; // nothing new
      lastReqKeyRef.current = key;

      setLoading(true);

      // cancel any in-flight call
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        signal: ctrl.signal,
      })
        .then((r) => r.json())
        .then((j) => {
          const arr = Array.isArray(j?.suggestions) ? j.suggestions : [];
          const cleaned = arr
            .filter((s) => typeof s === "string")
            .map((s) => s.trim())
            .filter((s) => s.length > 0)
            .slice(0, 3);
          setSugs(cleaned);
        })
        .catch(() => setSugs([]))
        .finally(() => setLoading(false));
    }, 400);

    return () => {
      clearTimeout(handle);
      // the next effect run will abort in-flight via abortRef
    };
  }, [body, endpoint, autoFetch]);

  if (loading || sugs.length === 0) return null;

  return (
    <div
      className={`w-full ${compact ? "mt-2" : "mt-3"} flex items-start gap-3`}
      aria-label="Ask GPT suggestions"
    >
      {/* Left arrow & label like ChatGPT */}
      <div className="flex items-center gap-2 text-sm text-zinc-300">
        <ArrowUturnLeftIcon className="h-4 w-4" />
        <span className="opacity-80">Ask GPT</span>
      </div>

      {/* Suggestion bubble */}
      <div className="flex-1">
        <div className="rounded-2xl bg-zinc-800/70 border border-zinc-700 px-4 py-3 shadow-sm">
          <div className="flex flex-wrap gap-2">
            {sugs.map((s, i) => (
              <button
                key={i}
                onClick={() => (onSend ? onSend(s) : onInsert?.(s))}
                className="text-zinc-50 text-sm rounded-full border border-zinc-600 px-3 py-1 hover:bg-zinc-700 active:scale-[0.99] transition"
                title={onSend ? "Send" : "Insert into input"}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Optional 'Yes' quick action */}
      {onSend && (
        <button
          onClick={() => onSend(sugs[0] || "Yes")}
          className="text-sm px-4 py-2 rounded-full bg-zinc-700 hover:bg-zinc-600 border border-zinc-500"
        >
          Yes
        </button>
      )}
    </div>
  );
}
