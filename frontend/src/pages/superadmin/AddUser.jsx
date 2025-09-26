// frontend/src/AddUser.jsx
import React, { useState } from "react";

export default function AddUser() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    role: "User",            // User | Admin | SuperAdmin | Client Support
    status: "active",        // active | inactive
    password: "",            // REQUIRED
  });
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [showPwd, setShowPwd] = useState(false);
  const [errors, setErrors] = useState({});

  const isStaff = ["Admin", "SuperAdmin", "Client Support"].includes(form.role);

  // simple validators
  const validate = () => {
    const e = {};
    if (!form.name.trim()) e.name = "Name is required";
    if (!form.email.trim()) e.email = "Email is required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) e.email = "Invalid email";
    if (!form.password) e.password = "Password is required";
    else if (form.password.length < 6) e.password = "Use at least 6 characters";
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  // normalize role names for backend
  const apiRole = () => {
    if (!isStaff) return "user";
    if (form.role === "Client Support") return "client_support";
    return form.role.toLowerCase(); // admin | superadmin
  };

  const endpoint = () => (isStaff ? "/api/admin/staff/users" : "/api/admin/users");

  const submit = async (e) => {
    e.preventDefault();
    setMsg("");
    if (!validate()) return;

    setSaving(true);
    try {
      const res = await fetch(endpoint(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          name: form.name.trim(),
          email: form.email.trim(),
          status: form.status,
          role: apiRole(),
          password: form.password, // ALWAYS send password
        }),
      });
      const j = await res.json();
      setSaving(false);

      if (!res.ok) {
        setMsg("❌ " + (j.error || "Failed to create user"));
        return;
      }

      setMsg(`✅ ${isStaff ? "Staff user" : "User"} created successfully.`);
      // Intentionally keep values (no reset), as requested.
    } catch (err) {
      setSaving(false);
      setMsg("❌ Network error, please try again.");
    }
  };

  return (
    <div className="w-full py-8 sm:py-10">
      {/* Center block horizontally */}
      <div className="max-w-xl mx-auto">
        <div className="rounded-2xl border bg-white shadow-sm p-6 sm:p-7">

          <h2 className="text-xl font-semibold">Add User</h2>

          <form className="grid gap-3 mt-5" onSubmit={submit} noValidate>
            <div>
              <label className="block text-sm mb-1">Full name</label>
              <input
                className="px-3 py-2 rounded-lg border w-full"
                placeholder="Enter Your Name"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
              {errors.name && <div className="text-xs text-red-600 mt-1">{errors.name}</div>}
            </div>

            <div>
              <label className="block text-sm mb-1">Email</label>
              <input
                className="px-3 py-2 rounded-lg border w-full"
                placeholder="Enter Your Email"
                type="email"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                autoComplete="email"
              />
              {errors.email && <div className="text-xs text-red-600 mt-1">{errors.email}</div>}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm mb-1">Role</label>
                <select
                  className="px-3 py-2 rounded-lg border w-full"
                  value={form.role}
                  onChange={(e) => setForm({ ...form, role: e.target.value })}
                >
                  <option>User</option>
                  <option>Admin</option>
                  <option>SuperAdmin</option>
                  <option>Client Support</option>
                </select>
              </div>

              <div>
                <label className="block text-sm mb-1">Status</label>
                <select
                  className="px-3 py-2 rounded-lg border w-full"
                  value={form.status}
                  onChange={(e) => setForm({ ...form, status: e.target.value })}
                >
                  <option value="active">active</option>
                  <option value="inactive">inactive</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm mb-1">Password (required)</label>
              <div className="relative">
                <input
                  type={showPwd ? "text" : "password"}
                  className="w-full px-3 py-2 rounded-lg border pr-20"
                  placeholder="Enter a secure password"
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-sm text-slate-600"
                  onClick={() => setShowPwd((s) => !s)}
                  aria-label={showPwd ? "Hide password" : "Show password"}
                >
                  {showPwd ? "Hide" : "Show"}
                </button>
              </div>
              {errors.password && <div className="text-xs text-red-600 mt-1">{errors.password}</div>}
            </div>

            <div className="flex items-center gap-3 pt-2">
              <button
                disabled={saving}
                className="px-4 py-2 rounded-lg bg-black text-white disabled:opacity-60"
              >
                {saving ? "Saving…" : "Create"}
              </button>
              {msg && <div className="text-sm">{msg}</div>}
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
