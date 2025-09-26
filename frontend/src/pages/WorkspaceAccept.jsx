import React, { useEffect, useState } from "react";

/**
 * Finalizes the invite.
 * - Reads ?token=… from the URL and POSTS it to /api/workspaces/invites/accept.
 * - If the API replies requires_login, we redirect to /login or /signup accordingly,
 *   including next=/workspace/accept?token=… so we finish after auth.
 */
export default function WorkspaceAccept() {
  const [msg, setMsg] = useState("Preparing…");

  useEffect(() => {
    (async () => {
      const params = new URLSearchParams(window.location.search);
      const token = params.get("token") || "";

      try {
        // Try to accept; when logged out, the API will tell us what to do next
        const r = await fetch("/api/workspaces/invites/accept", {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token }), // <-- important: pass token explicitly
        });
        const j = await r.json().catch(() => ({}));

        // Not authenticated yet → route to login or signup
        if (j?.requires_login) {
          const next = encodeURIComponent(`/workspace/accept?token=${encodeURIComponent(token)}`);
          if (j.has_account) {
            window.location.replace(`/login?next=${next}`);
          } else {
            const email = j.invite_email ? `&email=${encodeURIComponent(j.invite_email)}` : "";
            window.location.replace(`/signup?next=${next}${email}`);
          }
          return;
        }

        if (!r.ok) {
          const code = j?.error || `Failed (${r.status})`;
          setMsg(
            code === "expired" ? "This invite link has expired." :
            code === "invalid_or_used" ? "Invite link is invalid or already used." :
            code === "email_mismatch" ? "You are signed in with a different email than the invite was sent to." :
            code === "seats_exhausted" ? "All seats are used for that workspace." :
            "Could not join workspace."
          );
          return;
        }

        // success!
        window.location.replace("/workspace");
      } catch {
        setMsg("Network error.");
      }
    })();
  }, []);

  return (
    <div className="min-h-screen grid place-items-center bg-[#0b1324] text-white">
      <div className="rounded-2xl border border-white/10 bg-white/5 px-6 py-4 text-sm">
        {msg}
      </div>
    </div>
  );
}
