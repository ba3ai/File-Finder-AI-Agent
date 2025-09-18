import React, { useEffect, useRef, useState } from "react";
import { FiSend, FiMic, FiMicOff } from "react-icons/fi";

export default function WelcomeScreen({ userInput, setUserInput, onSend, onNewChat }) {
  const inputRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  // Optional voice (Chrome)
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
      if (t) {
        setUserInput(t);
        setTimeout(() => onSend(t), 150);
      }
    };
    recognitionRef.current = rec;
  }, [onSend, setUserInput]);

  const toggleVoice = () => {
    const r = recognitionRef.current;
    if (!r) return;
    try { isListening ? r.stop() : r.start(); } catch {}
  };

  const QuickChip = ({ text }) => (
    <button
      onClick={() => onSend(text)}
      className="px-4 py-2 rounded-full text-sm border border-[#22345e] bg-[#182544] text-white hover:bg-[#1d2a51] transition"
    >
      {text}
    </button>
  );

  return (
    <section className="flex min-h-screen items-center justify-center bg-[#0b1324] text-white px-4">
      <div className="w-full max-w-[980px] text-center">
        <h1 className="font-bold text-[clamp(32px,4.8vw,56px)] mb-6">Ready to dive in?</h1>

        {/* Copilot-like search bar */}
        <div className="mx-auto flex items-center gap-2 rounded-[22px] border border-[#1f2b4a] bg-gradient-to-b from-[#0f1a33] to-[#0d162b] px-3 py-2 shadow-[0_10px_30px_rgba(0,0,0,0.35)]">
          <input
            ref={inputRef}
            type="text"
            placeholder="Type here..."
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onSend(userInput)}
            className="flex-1 bg-transparent outline-none placeholder:text-[#9aa4bf] text-[18px] py-2 px-2"
          />
          <button
            onClick={toggleVoice}
            title={isListening ? "Stop voice input" : "Start voice input"}
            className={`grid place-items-center w-10 h-10 rounded-full border border-[#24355f] bg-[#132042] ${
              isListening ? "text-[#BD945B] animate-pulse" : "text-white"
            }`}
          >
            {isListening ? <FiMicOff size={18} /> : <FiMic size={18} />}
          </button>
          <button
            onClick={() => onSend(userInput)}
            title="Send"
            className="grid place-items-center w-10 h-10 rounded-full bg-[#E0C389] text-[#1b1720]"
          >
            <FiSend size={18} />
          </button>
        </div>

        {/* Quick suggestions */}
        <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
          <QuickChip text="Find the 2025 Pike valuation file" />
          <QuickChip text="Search SharePoint for quarterly financials" />
          <QuickChip text="List recent files I worked on in OneDrive" />
          <QuickChip text="Send me the latest HR policy PDF" />
        </div>

        <div className="mt-6 text-xs text-zinc-400">
          <button onClick={onNewChat} className="underline underline-offset-4">Start a fresh chat</button>
        </div>
      </div>
    </section>
  );
}
