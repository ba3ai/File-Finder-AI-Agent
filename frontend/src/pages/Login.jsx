// src/pages/Login.jsx
import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Eye, EyeOff, Mail, Lock } from "lucide-react";
import { motion } from "framer-motion";
import { FaFacebook } from "react-icons/fa";
import { FcGoogle } from "react-icons/fc";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        cache: "no-store",
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.error || "Login failed");

      // Clear any stale client-side state from a previous session
      try {
        sessionStorage.clear();
        localStorage.clear();
      } catch {}

      // Ensure the new session is active (and bypass any cache)
      await fetch("/api/auth/me", {
        credentials: "include",
        cache: "no-store",
        headers: { "Cache-Control": "no-cache" },
      });

      // Hard replace so the app remounts under the new identity
      window.location.replace("/dashboard");
    } catch (err) {
      setError(err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  const onForgot = async () => {
    if (!email) {
      setError("Enter your email first to receive a reset link.");
      return;
    }
    try {
      setError("");
      const res = await fetch(`${API_BASE}/api/auth/forgot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        cache: "no-store",
        body: JSON.stringify({ email }),
      });
      if (!res.ok) throw new Error("Could not send reset link");
      alert("If this email exists, a reset link has been sent.");
    } catch (e) {
      setError(e.message);
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
        <h1 className="text-3xl font-bold text-gray-900">Login</h1>

        <form onSubmit={onSubmit} className="mt-8 space-y-5">
          {error ? (
            <div className="p-3 rounded-lg bg-red-50 text-red-700 text-sm">{error}</div>
          ) : null}

          {/* Email */}
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

          {/* Password */}
          <label className="block">
            <span className="sr-only">Password</span>
            <div className="relative">
              <Lock className="w-5 h-5 absolute left-3 top-3.5 text-gray-400" />
              <input
                type={showPass ? "text" : "password"}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
                className="w-full pl-10 pr-11 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-4 focus:ring-blue-100 focus:border-blue-500 text-gray-900 placeholder:text-gray-400"
              />
              <button
                type="button"
                onClick={() => setShowPass((s) => !s)}
                aria-label={showPass ? "Hide password" : "Show password"}
                className="absolute right-3 top-2.5 p-1 text-gray-500 hover:text-gray-700"
              >
                {showPass ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>
          </label>

          <div className="flex items-center justify-between -mt-1">
            <button
              type="button"
              onClick={onForgot}
              className="text-sm text-blue-600 hover:text-blue-700"
            >
              Forgot password?
            </button>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl bg-blue-600 text-white font-semibold hover:bg-blue-700 transition shadow-md disabled:opacity-60"
          >
            {loading ? "Logging in..." : "Login"}
          </button>

          <p className="text-center text-sm text-gray-600">
            Don’t have an account?{" "}
            <Link to="/signup" className="text-blue-600 hover:text-blue-700 font-medium">
              Signup
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
