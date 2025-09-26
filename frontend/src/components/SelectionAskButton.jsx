// src/components/SelectionAskButton.jsx
import React, { useEffect, useRef, useState } from "react";

/**
 * Floating "Ask GPT" button that appears when the user selects text
 * inside the chat transcript. On click, it sets a quote block in the
 * composer. (No prefilled prompt by default.)
 *
 * Emits:
 *  - chat:set-quote  { text }
 *  - (optional) chat:insert-text { text } when prefillInstruction=true
 */
export default function SelectionAskButton({
  containerSelector = "#chat-transcript",
  maxSelection = 1200,
  prefillInstruction = false,         // <- default OFF
  instructionText = "Explain this clearly:", // used only if prefillInstruction is true
  label = "Ask Agent",
}) {
  const [visible, setVisible] = useState(false);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const selTextRef = useRef("");

  useEffect(() => {
    let raf = null;
    let lastText = "";

    const withinContainer = (range) => {
      try {
        const container = document.querySelector(containerSelector);
        if (!container || !range) return false;
        const common = range.commonAncestorContainer;
        const node = common?.nodeType === 1 ? common : common?.parentNode;
        return container.contains(node);
      } catch {
        return false;
      }
    };

    const update = () => {
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) {
        setVisible(false);
        lastText = "";
        return;
      }
      const text = (sel.toString() || "").trim();
      if (!text || text.length < 2 || text.length > maxSelection) {
        setVisible(false);
        lastText = "";
        return;
      }
      const range = sel.getRangeAt(0);
      if (!withinContainer(range)) {
        setVisible(false);
        lastText = "";
        return;
      }
      if (text === lastText) return;
      lastText = text;
      selTextRef.current = text;

      const rect = range.getBoundingClientRect();
      const x = Math.max(12, Math.min(window.innerWidth - 140, rect.left + rect.width / 2 - 50));
      const y = Math.max(12, rect.top - 36);
      setPos({ x, y: Math.max(8, y) });
      setVisible(true);
    };

    const onSelection = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(update);
    };

    document.addEventListener("selectionchange", onSelection);
    document.addEventListener("mouseup", onSelection);
    document.addEventListener("keyup", onSelection);

    return () => {
      if (raf) cancelAnimationFrame(raf);
      document.removeEventListener("selectionchange", onSelection);
      document.removeEventListener("mouseup", onSelection);
      document.removeEventListener("keyup", onSelection);
    };
  }, [containerSelector, maxSelection]);

  if (!visible) return null;

  const onAsk = () => {
    const t = selTextRef.current || "";
    if (!t) return;

    // Tell the composer to show a pretty quote block
    window.dispatchEvent(new CustomEvent("chat:set-quote", { detail: { text: t } }));

    // (Optional) Pre-fill an instruction only when explicitly enabled
    if (prefillInstruction && instructionText) {
      window.dispatchEvent(new CustomEvent("chat:insert-text", { detail: { text: instructionText } }));
    }

    // Clear selection and hide the pill
    try { window.getSelection()?.removeAllRanges(); } catch {}
    setVisible(false);
  };

  return (
    <button
      onClick={onAsk}
      style={{ position: "fixed", left: pos.x, top: pos.y, zIndex: 9999 }}
      className="rounded-full bg-zinc-800/95 text-zinc-100 border border-zinc-700 px-3 py-1 text-xs shadow-lg hover:bg-zinc-700"
    >
      {label}
    </button>
  );
}
