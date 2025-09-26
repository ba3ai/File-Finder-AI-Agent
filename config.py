import os

class Config:
    # 2Checkout (Verifone)
    TWOCHECKOUT_SELLER_ID = os.getenv("TWOCHECKOUT_SELLER_ID", "")
    TWOCHECKOUT_SECRET    = os.getenv("TWOCHECKOUT_SECRET", "")
    TWOCHECKOUT_SANDBOX   = os.getenv("TWOCHECKOUT_SANDBOX", "true").lower() == "true"
    TWOCHECKOUT_BUY_URL   = os.getenv("TWOCHECKOUT_BUY_URL", "https://sandbox.2checkout.com/checkout/buy")
    FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:5000")
    # Team features test switch
    TEAM_FEATURES_DISABLED = os.getenv("TEAM_FEATURES_DISABLED", "True").lower() in ("1","true","yes")
    TEAM_DEV_SEATS_LIMIT   = int(os.getenv("TEAM_DEV_SEATS_LIMIT", "50"))  # optional big seat cap while testing
    


    # ✅ Add your plans here (Pro included). Codes must match 2Checkout product codes.
    BILLING_PLANS = [
    # … your existing Starter/Pro …
    {
        "code": "FF_TEAM_M",
        "name": "Team",
        "interval": "month",
        "price": 59,
        "currency": "USD",
        "features": [
            "All Pro features", "Shared workspace", "Team seats (3 included)"
        ],
        "highlight": True
    },
    {
        "code": "FF_TEAM_Y",
        "name": "Team",
        "interval": "year",
        "price": 590,
        "currency": "USD",
        "features": [
            "All Pro features", "Shared workspace", "Team seats (3 included)"
        ],
        "highlight": True
    },
]


