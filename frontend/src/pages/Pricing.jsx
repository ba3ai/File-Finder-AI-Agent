import React, { useEffect, useMemo, useState } from "react";

export default function Pricing() {
  const [plans, setPlans] = useState([]);         // unified plan list
  const [period, setPeriod] = useState("monthly"); // "monthly" | "yearly"
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  // Normalize API responses (supports both new 2CO and legacy Stripe responses)
  const normalizePlans = (d) => {
    // New 2Checkout shape
    if (Array.isArray(d?.plans) && d?.provider === "2checkout") {
      return d.plans.map(p => ({
        code: p.code,
        name: p.name,
        interval: p.interval, // "month" | "year"
        price: p.price,       // number
        currency: p.currency || "USD",
        features: Array.isArray(p.features) ? p.features : [],
      }));
    }
    // Legacy Stripe shape (convert to 2CO-like)
    if (Array.isArray(d?.plans)) {
      return d.plans.flatMap(p => {
        const base = {
          code: p.code || p.id || p.name,
          name: p.name,
          features: Array.isArray(p.features) ? p.features : [],
          currency: p.currency || "USD",
        };
        const out = [];
        if (p.monthly_price) {
          out.push({ ...base, interval: "month", price: p.monthly_price });
        }
        if (p.yearly_price) {
          out.push({ ...base, interval: "year", price: p.yearly_price });
        }
        // Fallback if no explicit amounts provided
        if (!out.length) {
          out.push({ ...base, interval: "month", price: p.price || 0 });
        }
        return out;
      });
    }
    return [];
  };

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setErr("");
    fetch("/api/billing/prices", { credentials: "include" })
      .then(r => r.json())
      .then(d => { if (!alive) return; setPlans(normalizePlans(d)); })
      .catch(e => { if (!alive) return; setErr(e.message || "Failed to load pricing"); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  const displayPlans = useMemo(() => {
    const want = period === "yearly" ? "year" : "month";
    return plans.filter(p => (p.interval || "month") === want);
  }, [plans, period]);

  const formatPrice = (n, currency = "USD", interval = "month") => {
    try {
      const s = new Intl.NumberFormat(undefined, { style: "currency", currency }).format(n);
      return `${s}/${interval === "year" ? "yr" : "mo"}`;
    } catch {
      return `$${n}/${interval === "year" ? "yr" : "mo"}`;
    }
  };

  const startCheckout = async (planCode) => {
    try {
      const res = await fetch("/api/billing/checkout", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan_code: planCode })
      });
      const data = await res.json();
      if (res.ok && data?.url) {
        window.location.href = data.url; // 2Checkout hosted checkout
      } else {
        throw new Error(data?.error || "Unable to start checkout");
      }
    } catch (e) {
      alert(e.message || "Checkout failed");
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="mx-auto max-w-6xl px-4 py-16">
        <h1 className="text-4xl font-semibold text-center">Choose your plan</h1>

        {/* Period toggle */}
        <div className="mt-8 flex items-center justify-center gap-4">
          <button
            onClick={() => setPeriod("monthly")}
            className={`px-4 py-2 rounded-full ${period==="monthly"?"bg-white text-black":"bg-gray-800"}`}
          >
            Monthly
          </button>
          <button
            onClick={() => setPeriod("yearly")}
            className={`px-4 py-2 rounded-full ${period==="yearly"?"bg-white text-black":"bg-gray-800"}`}
          >
            Yearly <span className="ml-1 text-sm opacity-70">(save vs monthly)</span>
          </button>
        </div>

        {/* Status */}
        {err && (
          <div className="mt-6 mx-auto max-w-xl rounded-xl bg-red-900/30 border border-red-800 p-3 text-sm text-red-200">
            {err}
          </div>
        )}

        {/* Cards */}
        <div className="grid md:grid-cols-3 gap-6 mt-10">
          {loading ? (
            Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="rounded-2xl bg-gray-900 border border-gray-800 p-6 shadow-xl animate-pulse">
                <div className="h-6 w-32 bg-gray-800 rounded" />
                <div className="mt-4 h-4 w-48 bg-gray-800 rounded" />
                <div className="mt-6 h-10 w-full bg-gray-800 rounded-xl" />
              </div>
            ))
          ) : displayPlans.length === 0 ? (
            <div className="col-span-full text-center text-gray-300">
              No plans available for this billing period.
            </div>
          ) : (
            displayPlans.map(p => (
              <div key={`${p.code}-${p.interval}`} className="rounded-2xl bg-gray-900 border border-gray-800 p-6 shadow-xl flex flex-col">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="text-xl font-semibold">{p.name}</h3>
                    <div className="mt-1 text-gray-300">{formatPrice(p.price ?? 0, p.currency, p.interval)}</div>
                  </div>
                  <span className="text-[10px] tracking-wider rounded-full px-3 py-1 bg-gray-800 border border-gray-700">
                    {p.code}
                  </span>
                </div>

                {Array.isArray(p.features) && p.features.length > 0 && (
                  <div className="mt-5">
                    <ul className="space-y-2 text-sm text-gray-300">
                      {p.features.map((f, i) => (
                        <li key={i} className="flex gap-2">
                          <span>•</span><span>{f}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <button
                  onClick={() => startCheckout(p.code)}
                  className="mt-6 w-full rounded-2xl bg-white text-black font-medium py-3 hover:opacity-90 transition"
                >
                  Get {p.name}
                </button>
              </div>
            ))
          )}
        </div>

        {/* FAQ */}
        <div className="mt-16 grid md:grid-cols-2 gap-8 text-gray-300">
          <div>
            <h4 className="text-lg font-semibold text-white">How do trials work?</h4>
            <p className="mt-2 text-sm">
              You keep full access until your trial end date. After that, pick a plan to continue.
            </p>
          </div>
          <div>
            <h4 className="text-lg font-semibold text-white">Can I cancel anytime?</h4>
            <p className="mt-2 text-sm">
              Yes—manage your subscription from the billing portal. Changes apply next billing cycle.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
