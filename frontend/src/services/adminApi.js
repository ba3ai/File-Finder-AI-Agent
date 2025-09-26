const BASE = import.meta.env.VITE_API_BASE || '';

async function http(path, opts={}) {
  const res = await fetch(`${BASE}${path}`, {
    credentials: 'include',
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' },
    ...opts,
  });
  const data = await res.json().catch(()=> ({}));
  if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
  return data;
}

export const adminApi = {
  users: {
    list: (params={}) => {
      const q = new URLSearchParams(params).toString();
      return http(`/api/admin/users${q?`?${q}`:''}`);
    },
    create: (payload) => http('/api/admin/users', { method: 'POST', body: JSON.stringify(payload) }),
    update: (id, payload) => http(`/api/admin/users/${id}`, { method: 'PATCH', body: JSON.stringify(payload) }),
  },
  subs: {
    list: (params={}) => {
      const q = new URLSearchParams(params).toString();
      return http(`/api/admin/subscriptions${q?`?${q}`:''}`);
    }
  },
  tickets: {
    list: (params={}) => {
      const q = new URLSearchParams(params).toString();
      return http(`/api/admin/tickets${q?`?${q}`:''}`);
    }
  },
  revenue: {
    summary: () => http(`/api/admin/revenue`)
  }
};
