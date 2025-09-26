export default function ManageBilling() {
  const openPortal = async () => {
    const res = await fetch("/api/billing/portal", {
      method: "POST",
      credentials: "include"
    });
    const data = await res.json();
    if (data.url) window.location.href = data.url;
    else alert(data.error || "No billing portal available");
  };
  return (
    <div className="p-6">
      <h1 className="text-2xl font-semibold">Billing</h1>
      <p className="mt-2 text-gray-500">Update payment method, download invoices, or cancel your plan.</p>
      <button onClick={openPortal} className="mt-6 rounded-xl bg-white text-black px-4 py-2 font-medium">
        Open Billing Portal
      </button>
    </div>
  );
}
