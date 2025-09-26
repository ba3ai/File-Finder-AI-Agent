// StaffLogin.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function StaffLogin() {
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    setMsg("");
    setLoading(true);
    try {
      const res = await fetch("/api/admin/staff/login", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const j = await res.json();
      setLoading(false);
      if (!res.ok) {
        setMsg(j.message || j.error || "Login failed");
        return;
      }
      // success → go to the superadmin dashboard
      nav(j.redirect || "/superadmin/overview", { replace: true });
    } catch {
      setLoading(false);
      setMsg("Network error. Please try again.");
    }
  };

  return (
    <div className="min-h-screen grid place-items-center bg-gradient-to-br from-slate-50 to-slate-100 px-4">
      <div className="w-full max-w-md bg-white border rounded-2xl shadow-sm p-6">
        <div className="mb-5">
          <h1 className="text-xl font-semibold">Staff Sign In</h1>
          <p className="text-sm text-slate-600 mt-1">
            Super Admin, Admin, and Client Support users can sign in here.
          </p>
        </div>

        <form onSubmit={submit} className="grid gap-3">
          <div>
            <label className="block text-sm mb-1">Email</label>
            <input
              type="email"
              className="w-full px-3 py-2 rounded-lg border"
              placeholder="you@company.com"
              value={email}
              onChange={(e)=>setEmail(e.target.value)}
              required
            />
          </div>

          <div>
            <label className="block text-sm mb-1">Password</label>
            <div className="relative">
              <input
                type={show ? "text" : "password"}
                className="w-full px-3 py-2 rounded-lg border pr-10"
                placeholder="••••••••"
                value={password}
                onChange={(e)=>setPassword(e.target.value)}
                required
              />
              <button
                type="button"
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 text-sm"
                onClick={()=>setShow(s=>!s)}
              >
                {show ? "Hide" : "Show"}
              </button>
            </div>
          </div>

        <button
          disabled={loading}
          className="mt-1 px-4 py-2 rounded-lg bg-black text-white"
        >
          {loading ? "Signing in…" : "Sign in"}
        </button>

          {msg && <div className="text-sm text-red-600">{msg}</div>}
        </form>
      </div>
    </div>
  );
}
