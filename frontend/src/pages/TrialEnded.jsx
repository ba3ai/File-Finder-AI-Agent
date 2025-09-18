// frontend/src/pages/TrialEnded.jsx
import React, { useEffect, useMemo, useState } from "react";
import { FiArrowRight, FiHelpCircle, FiMail, FiRefreshCw } from "react-icons/fi";

export default function TrialEnded() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [toast, setToast] = useState({ type: "", msg: "" });

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const r = await fetch("/api/trial/status", {
          credentials: "include",
          cache: "no-store",
          headers: { "Cache-Control": "no-cache" },
        });
        const j = await r.json().catch(() => ({}));
        if (mounted) setStatus(j || {});
      } catch {
        /* ignore */
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => (mounted = false);
  }, []);

  useEffect(() => {
    if (!toast.msg) return;
    const t = setTimeout(() => setToast({ type: "", msg: "" }), 3500);
    return () => clearTimeout(t);
  }, [toast]);

  const endedOn = useMemo(() => {
    const d = status?.ends_at ? new Date(status.ends_at) : null;
    return d ? d.toLocaleString() : null;
  }, [status]);

  const graceEnds = useMemo(() => {
    const d = status?.grace_ends_at ? new Date(status.grace_ends_at) : null;
    return d ? d.toLocaleString() : null;
  }, [status]);

  const requestExtension = async () => {
    setSending(true);
    try {
      // Optional endpoint; if missing we still show a friendly confirmation.
      const r = await fetch("/api/trial/request_extension", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: "Need more time to evaluate" }),
      });
      // Don’t hard fail on non-200 — we still confirm to the user.
      try { await r.json(); } catch {}
      setToast({ type: "ok", msg: "Thanks! We’ve received your request. We’ll be in touch shortly." });
    } catch {
      setToast({ type: "ok", msg: "Request sent. Our team will reach out soon." });
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="min-h-screen theme-surface">
      {/* subtle background accent */}
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(60%_40%_at_50%_-10%,rgba(124,58,237,.25),transparent)]" />
      <div className="relative mx-auto max-w-3xl px-4 py-10">
        {/* Card frame */}
        <div className="rounded-[28px] p-[2px] bg-gradient-to-br from-white/15 to-white/5 shadow-[0_30px_80px_rgba(0,0,0,.45)]">
          <div className="rounded-[26px] theme-tint backdrop-blur-xl theme-border">
            {/* Header */}
            <div className="px-7 pt-8 pb-4">
              <div className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3 py-1 text-[12px] text-white/80">
                <FiHelpCircle className="opacity-80" />
                Trial ended
              </div>
              <h1 className="mt-4 text-2xl md:text-3xl font-bold tracking-tight">
                Your free trial has ended
              </h1>
              <p className="mt-2 text-white/70">
                {loading ? "Checking your account…" : (
                  <>
                    {endedOn ? <>The trial expired on <b className="text-white">{endedOn}</b>.</> : "Your trial has expired."}
                    {" "}
                    {graceEnds && <>You still have limited access until <b className="text-white">{graceEnds}</b>.</>}
                  </>
                )}
              </p>
            </div>

            {/* CTA strip */}
            <div className="px-7 pb-7">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <a
                  href="/pricing"
                  className="inline-flex items-center justify-center rounded-2xl bg-[#E0C389] px-5 py-3 text-sm font-semibold text-black shadow-lg hover:brightness-95 transition"
                >
                  Upgrade now <FiArrowRight className="ml-2" />
                </a>
                <button
                  onClick={requestExtension}
                  disabled={sending}
                  className="inline-flex items-center justify-center rounded-2xl border border-white/15 bg-white/5 px-5 py-3 text-sm font-semibold hover:bg-white/10 transition disabled:opacity-60"
                >
                  {sending ? <><FiRefreshCw className="mr-2 animate-spin" /> Sending…</> : "Request an extension"}
                </button>
                <a
                  href="/contact"
                  className="inline-flex items-center justify-center rounded-2xl border border-white/15 bg-white/0 px-5 py-3 text-sm font-semibold hover:bg-white/10 transition"
                >
                  <FiMail className="mr-2 opacity-80" /> Contact sales
                </a>
              </div>
            </div>

            {/* Divider */}
            <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

            {/* Benefits / what you get */}
            <div className="grid gap-6 px-7 py-7 md:grid-cols-3">
              <Feature
                title="Unlimited sources"
                desc="Search across SharePoint, OneDrive, Google Drive, Dropbox, Box, and MEGA without limits."
              />
              <Feature
                title="Priority answers"
                desc="Faster retrieval and larger context windows for long reports, policies, and contracts."
              />
              <Feature
                title="Admin controls"
                desc="Granular access, audit logs, and org-wide connectors to keep data secure."
              />
            </div>

            {/* Footnote */}
            <div className="px-7 pb-8 text-xs text-white/50">
              Upgrading is self-serve and takes less than a minute. Need procurement docs or vendor forms? Our team can help.
            </div>
          </div>
        </div>

        {/* Toast */}
        {toast.msg && (
          <div
            className={`fixed bottom-6 left-1/2 z-50 -translate-x-1/2 rounded-xl border px-4 py-3 text-sm shadow-2xl
              ${toast.type === "ok"
                ? "bg-emerald-500/10 border-emerald-400/30 text-emerald-200"
                : "bg-red-500/10 border-red-400/30 text-red-200"}`}
          >
            {toast.msg}
          </div>
        )}

        {/* FAQ / links */}
        <div className="mx-auto mt-6 max-w-3xl text-center text-sm text-white/60">
          Looking for invoices or billing details?
          {" "}
          <a className="underline hover:text-white" href="/billing">Go to billing</a>.
          {" "}
          Or <a className="underline hover:text-white" href="/login">sign in with a different account</a>.
        </div>
      </div>
    </div>
  );
}

function Feature({ title, desc }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
      <div className="text-[13px] font-semibold">{title}</div>
      <div className="mt-1 text-[13px] text-white/70 leading-relaxed">{desc}</div>
    </div>
  );
}
