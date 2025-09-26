export default function SubscribeButton({ planCode, priceId, children }) {
  const go = async () => {
    const res = await fetch("/api/billing/checkout", {
      method: "POST",
      credentials: "include",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ plan_code: planCode, price_id: priceId })
    });
    const data = await res.json();
    if (data.url) window.location.href = data.url;
    else alert(data.error || "Unable to start checkout");
  };
  return (
    <button onClick={go} className="rounded-xl bg-white text-black px-4 py-2 font-medium">
      {children || "Subscribe"}
    </button>
  );
}
