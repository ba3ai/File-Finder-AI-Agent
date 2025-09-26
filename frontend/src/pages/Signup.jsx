// src/pages/Signup.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff, Mail, Lock } from "lucide-react";
import { motion } from "framer-motion";
import { FaFacebook } from "react-icons/fa";
import { FcGoogle } from "react-icons/fc";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export default function Signup() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [show1, setShow1] = useState(false);
  const [show2, setShow2] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // honor ?next= so invite flow resumes after auth
  const next = useMemo(() => {
    const p = new URLSearchParams(window.location.search);
    const n = p.get("next");
    return n && n.startsWith("/") ? n : "/";
  }, []);

  // prefill email if ?email=foo@bar was passed from the invite
  useEffect(() => {
    const p = new URLSearchParams(window.location.search);
    const e = p.get("email");
    if (e) setEmail(e);
  }, []);

  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.error || "Signup failed");

      // Some backends auto-login after register; check if that happened.
      const me = await fetch(`/api/auth/me`, {
        credentials: "include",
        cache: "no-store",
        headers: { "Cache-Control": "no-cache" },
      }).then(r => r.json()).catch(() => ({}));

      if (me?.ok || me?.email || me?.logged_in) {
        // Already authenticated → continue to next (invite accept page).
        window.location.replace(next);
        return;
      }

      // Not logged in yet → bounce to login carrying next+email so flow resumes.
      const q = new URLSearchParams({ next, email }).toString();
      window.location.replace(`/login?${q}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-blue-600 flex items-center justify-center px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        className="bg-white w-full max-w-lg rounded-2xl shadow-xl p-8 md:p-10"
      >
        <h1 className="text-3xl font-bold text-gray-900">Signup</h1>

        <form onSubmit={onSubmit} className="mt-8 space-y-5">
          {error ? (
            <div className="p-3 rounded-lg bg-red-50 text-red-700 text-sm">{error}</div>
          ) : null}

          <label className="block">
            <span className="sr-only">Email</span>
            <div className="relative">
              <Mail className="w-5 h-5 absolute left-3 top-3.5 text-gray-400" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Email"
                className="w-full pl-10 pr-4 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-4 focus:ring-blue-100 focus:border-blue-500 text-gray-900 placeholder:text-gray-400"
              />
            </div>
          </label>

          <label className="block">
            <span className="sr-only">Create password</span>
            <div className="relative">
              <Lock className="w-5 h-5 absolute left-3 top-3.5 text-gray-400" />
              <input
                type={show1 ? "text" : "password"}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Create password"
                className="w-full pl-10 pr-11 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-4 focus:ring-blue-100 focus:border-blue-500 text-gray-900 placeholder:text-gray-400"
              />
              <button
                type="button"
                onClick={() => setShow1((s) => !s)}
                aria-label={show1 ? "Hide password" : "Show password"}
                className="absolute right-3 top-2.5 p-1 text-gray-500 hover:text-gray-700"
              >
                {show1 ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>
          </label>

          <label className="block">
            <span className="sr-only">Confirm password</span>
            <div className="relative">
              <Lock className="w-5 h-5 absolute left-3 top-3.5 text-gray-400" />
              <input
                type={show2 ? "text" : "password"}
                required
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                placeholder="Confirm password"
                className="w-full pl-10 pr-11 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-4 focus:ring-blue-100 focus:border-blue-500 text-gray-900 placeholder:text-gray-400"
              />
              <button
                type="button"
                onClick={() => setShow2((s) => !s)}
                aria-label={show2 ? "Hide password" : "Show password"}
                className="absolute right-3 top-2.5 p-1 text-gray-500 hover:text-gray-700"
              >
                {show2 ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>
          </label>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl bg-blue-600 text-white font-semibold hover:bg-blue-700 transition shadow-md disabled:opacity-60"
          >
            {loading ? "Creating account..." : "Signup"}
          </button>

          <p className="text-center text-sm text-gray-600">
            Already have an account?{" "}
            <Link to={`/login?next=${encodeURIComponent(next)}${email ? `&email=${encodeURIComponent(email)}` : ""}`} className="text-blue-600 hover:text-blue-700 font-medium">
              Login
            </Link>
          </p>

          <Divider text="Or" />

          <div className="space-y-3">
            <button
              type="button"
              className="w-full py-3 rounded-xl bg-[#4267B2] text-white font-semibold flex items-center justify-center gap-3 hover:brightness-95"
            >
              <FaFacebook className="text-xl" /> Login with Facebook
            </button>
            <button
              type="button"
              disabled
              className="w-full py-3 rounded-xl bg-gray-100 text-gray-600 font-semibold flex items-center justify-center gap-3 cursor-not-allowed"
            >
              <FcGoogle className="text-xl" /> Login with Google
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  );
}

function Divider({ text }) {
  return (
    <div className="flex items-center gap-4">
      <span className="h-px flex-1 bg-gray-200" />
      <span className="text-gray-400 text-sm">{text}</span>
      <span className="h-px flex-1 bg-gray-200" />
    </div>
  );
}
