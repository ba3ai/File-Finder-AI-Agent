# admin_seed.py
from datetime import datetime, timedelta
from app import app  # assuming your app/db are initialized here
from models import db, User
from admin_models import Subscription, Ticket, Revenue

with app.app_context():
    db.create_all()  # will create new tables if missing

    # Ensure a couple of users exist (adjust to your actual users)
    u1 = User.query.filter_by(email="alice@ba3.ai").first()
    if not u1:
        u1 = User(name="Alice Rahman", email="alice@ba3.ai", status="active")
        db.session.add(u1)

    u2 = User.query.filter_by(email="muzahid@ba3.ai").first()
    if not u2:
        u2 = User(name="Muzahid Hasan", email="muzahid@ba3.ai", status="active")
        db.session.add(u2)
    db.session.commit()

    # Subscriptions
    if Subscription.query.count() == 0:
        db.session.add_all([
            Subscription(user_id=u1.id, plan="Pro", price=49, start_date=datetime.utcnow()-timedelta(days=60), end_date=datetime.utcnow()+timedelta(days=30), status="active"),
            Subscription(user_id=u2.id, plan="Starter", price=19, start_date=datetime.utcnow()-timedelta(days=20), end_date=datetime.utcnow()+timedelta(days=10), status="active"),
        ])

    # Tickets
    if Ticket.query.count() == 0:
        db.session.add_all([
            Ticket(title="Onboard new investor group", priority="high", status="open", assigned_to=u2.id),
            Ticket(title="Fix OAuth redirect", priority="high", status="in_progress", assigned_to=u1.id),
            Ticket(title="Add billing export CSV", priority="medium", status="closed", assigned_to=u1.id),
        ])

    # Revenue
    if Revenue.query.count() == 0:
        now = datetime.utcnow()
        for i in range(12):
            dt = now - timedelta(days=30*i)
            db.session.add(Revenue(
                invoice_no=f"INV-{202500+i:05d}",
                amount=1000 + 150*i,
                status=("paid" if i % 4 != 0 else "due"),
                invoice_date=dt
            ))

    db.session.commit()
    print("Admin seed complete.")
