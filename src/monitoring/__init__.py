"""Monitoring: store, dashboard, staleness, Telegram, scheduler.

Import submodules directly (e.g. ``from src.monitoring.dashboard import ...``).
Do not re-export scheduler here: it imports ``scripts.monitoring`` and breaks
read-only images (business console) that only need dashboard/store APIs.
"""
