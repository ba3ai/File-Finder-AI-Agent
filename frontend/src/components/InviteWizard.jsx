import React, { useMemo, useState } from "react";

const roles = ["Member", "Admin", "Owner"];
const norm = (s = "") => s.trim().toLowerCase();
const isEmail = (s) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(s);

export default function InviteWizard({ onClose }) {
  const [raw, setRaw] = useState("");
  const [chips, setChips] = useState([]); // [{ email, role }]
  const [step, setStep] = useState(1);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [note, setNote] = useState("");

  const addChip = (email) => {
    const e = norm(email);
    if (!isEmail(e)) return;
    if (chips.some((c) => c.email === e)) return;
    setChips([...chips, { email: e, role: "Member" }]);
  };

  const removeChip = (email) =>
    setChips(chips.filter((c) => c.email !== email));

  const onRawKeyDown = (e) => {
    if (e.key === "Enter" || e.key === "," || e.key === " ") {
      e.preventDefault();
      const text = raw.replace(/[,\s]+$/, "");
      if (text) addChip(text);
      setRaw("");
    } else if (e.key === "Backspace" && !raw) {
      setChips(chips.slice(0, -1));
    }
  };

  const parsed = useMemo(() => chips, [chips]);

  const send = async () => {
    if (!parsed.length) {
      setErr("Add at least one valid email.");
      return;
    }
    setBusy(true);
    setErr("");
    setNote("");
    try {
      const r = await fetch("/api/workspaces/invites/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          invites: parsed.map((c) => ({
            email: c.email,
            role: c.role.toLowerCase(),
          })),
        }),
      });
      const j = await r.json();
      if (!r.ok) {
        setErr(j.error || "Failed to send invites.");
        return;
      }
      setNote(`Sent ${parsed.length} invite${parsed.length > 1 ? "s" : ""}.`);
      setStep(3);
    } catch {
      setErr("Network error.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/50 p-4">
      <div className="w-full max-w-2xl rounded-2xl border border-white/10 bg-[#111827] text-white shadow-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-white/10 text-lg font-semibold">
          {step === 1 && "Invite by email"}
          {step === 2 && "Invite members to your workspace"}
          {step === 3 && "Invitations sent"}
        </div>

        {step === 1 && (
          <div className="p-5">
            <div className="text-sm text-white/80 mb-2">Emails</div>
            <div className="rounded-xl border border-white/15 bg-[#0b1324] p-2 min-h-[120px]">
              <div className="flex flex-wrap gap-2">
                {chips.map((c) => (
                  <span
                    key={c.email}
                    className="inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1"
                  >
                    <span className="text-sm">{c.email}</span>
                    <button
                      className="text-white/70 hover:text-white"
                      onClick={() => removeChip(c.email)}
                    >
                      ✕
                    </button>
                  </span>
                ))}
                <input
                  value={raw}
                  onChange={(e) => setRaw(e.target.value)}
                  onKeyDown={onRawKeyDown}
                  placeholder="type email and press Enter"
                  className="flex-1 min-w-[180px] bg-transparent outline-none text-sm placeholder-white/40"
                />
              </div>
            </div>

            <div className="mt-4 rounded-xl border border-white/10 bg-white/5 p-3 text-sm text-white/70">
              <div className="flex items-start gap-2">
                <span className="mt-0.5 inline-block h-2 w-2 rounded-full bg-white/40" />
                <div>
                  Users that accept invites will be included as additional seats
                  on your next invoice.
                </div>
              </div>
            </div>

            {err && <div className="mt-3 text-sm text-red-300">{err}</div>}

            <div className="mt-5 flex items-center justify-between">
              <button
                className="rounded-xl px-4 py-2 text-sm border border-white/15 hover:bg-white/5"
                onClick={onClose}
              >
                Skip
              </button>
              <button
                className="rounded-xl px-5 py-2 text-sm font-medium bg-emerald-500 text-black hover:opacity-90 disabled:opacity-50"
                disabled={!chips.length}
                onClick={() => setStep(2)}
              >
                Next
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="p-5">
            <div className="text-white/80 text-sm mb-3">
              You can invite members to join the workspace. You can do so at any
              time from the workspace settings page.
            </div>

            <div className="rounded-xl border border-white/10 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-white/5">
                  <tr className="text-white/70">
                    <th className="p-3 text-left">Email</th>
                    <th className="p-3 text-left">Role</th>
                    <th className="p-3 w-16" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {parsed.map((c, idx) => (
                    <tr key={c.email}>
                      <td className="p-3">{c.email}</td>
                      <td className="p-3">
                        <select
                          className="bg-[#0b1324] border border-white/15 rounded-lg px-2 py-1"
                          value={c.role}
                          onChange={(e) => {
                            const v = [...chips];
                            v[idx] = { ...v[idx], role: e.target.value };
                            setChips(v);
                          }}
                        >
                          {roles.map((r) => (
                            <option key={r} value={r}>
                              {r}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="p-3 text-right">
                        <button
                          className="text-white/70 hover:text-white"
                          onClick={() => removeChip(c.email)}
                          title="Remove"
                        >
                          🗑
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {err && <div className="mt-3 text-sm text-red-300">{err}</div>}

            <div className="mt-5 flex items-center justify-between">
              <button
                className="rounded-xl px-4 py-2 text-sm border border-white/15 hover:bg-white/5"
                onClick={() => setStep(1)}
              >
                Back
              </button>
              <button
                className="rounded-xl px-5 py-2 text-sm font-medium bg-emerald-500 text-black hover:opacity-90 disabled:opacity-50"
                disabled={!parsed.length || busy}
                onClick={send}
              >
                {busy ? "Sending invites…" : "Send invites"}
              </button>
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="p-5">
            <div className="text-emerald-300 text-sm">
              {note || "Invitations sent."}
            </div>
            <div className="mt-4 text-right">
              <button
                className="rounded-xl px-5 py-2 text-sm border border-white/15 hover:bg-white/5"
                onClick={onClose}
              >
                Close
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
