-- Users table assumed to exist with at least: id, email, trial_ends_at (ISO string), created_at
-- Add minimal subscription tables.

CREATE TABLE IF NOT EXISTS subscriptions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  stripe_customer_id TEXT,
  stripe_subscription_id TEXT,
  plan_id TEXT,                -- your internal plan code e.g. STARTER, PRO, ENTERPRISE
  price_id TEXT,               -- Stripe price id e.g. price_ABC
  status TEXT,                 -- trialing, active, past_due, canceled, incomplete, unpaid, paused
  current_period_start TEXT,
  current_period_end TEXT,
  cancel_at_period_end INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS invoices (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  stripe_invoice_id TEXT,
  amount_due INTEGER,
  amount_paid INTEGER,
  currency TEXT,
  hosted_invoice_url TEXT,
  status TEXT,                 -- draft, open, paid, void, uncollectible
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS webhook_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  stripe_event_id TEXT UNIQUE,
  type TEXT,
  payload_json TEXT,
  received_at TEXT DEFAULT (datetime('now'))
);
