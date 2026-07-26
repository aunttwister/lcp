"""Budget manager — CRUD operations and spend enforcement.

Tracks spending against budgets defined per key and/or profile.
Supports monthly and total budgets with configurable thresholds and actions.
"""

from datetime import datetime, timezone

from .logging_config import get_logger
from .models import Budget, get_session

logger = get_logger("lcp.budgets")


class BudgetManager:
    """Manages spending budgets with DB persistence and enforcement."""

    def __init__(self, engine):
        self._engine = engine

    # ── CRUD ──────────────────────────────────────────────────────────────

    def list_budgets(self) -> list[dict]:
        """List all budgets."""
        with get_session(self._engine) as session:
            budgets = session.query(Budget).order_by(Budget.id.desc()).all()
            return [
                {
                    "id": b.id,
                    "name": b.name,
                    "key_id": b.key_id,
                    "profile": b.profile,
                    "amount": b.amount,
                    "current_spend": b.current_spend,
                    "period": b.period,
                    "threshold_pct": b.threshold_pct,
                    "action": b.action,
                    "status": b.status,
                    "created_at": b.created_at,
                    "last_alert_at": b.last_alert_at,
                    "spend_pct": round((b.current_spend / b.amount * 100), 1) if b.amount > 0 else 0,
                }
                for b in budgets
            ]

    def create_budget(
        self,
        name: str,
        amount: float,
        key_id: int | None = None,
        profile: str | None = None,
        period: str = "monthly",
        threshold_pct: str = "50,80,90",
        action: str = "log",
    ) -> dict:
        """Create a new budget."""
        with get_session(self._engine) as session:
            b = Budget(
                name=name,
                key_id=key_id,
                profile=profile or None,
                amount=amount,
                period=period,
                threshold_pct=threshold_pct,
                action=action,
                status="active",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            session.add(b)
            session.commit()
            budget_id = b.id

        logger.info("budget_created", id=budget_id, name=name, amount=amount)
        return {
            "ok": True,
            "id": budget_id,
            "name": name,
            "amount": amount,
        }

    def update_budget(self, budget_id: int, updates: dict) -> dict | None:
        """Update budget fields. Returns updated budget or None if not found."""
        with get_session(self._engine) as session:
            b = session.query(Budget).filter(Budget.id == budget_id).first()
            if not b:
                return None
            for field in ("name", "amount", "period", "threshold_pct", "action", "status"):
                if field in updates:
                    setattr(b, field, updates[field])
            session.commit()
            return {"ok": True, "id": budget_id}

    def delete_budget(self, budget_id: int) -> bool:
        """Delete a budget by ID."""
        with get_session(self._engine) as session:
            b = session.query(Budget).filter(Budget.id == budget_id).first()
            if not b:
                return False
            session.delete(b)
            session.commit()
        logger.info("budget_deleted", id=budget_id)
        return True

    def get_budget_status(self) -> list[dict]:
        """Get current budget status for all active budgets with threshold breaches."""
        return self.list_budgets()

    # ── Spend Tracking & Enforcement ──────────────────────────────────────

    def record_spend(self, cost: float, key_id: int | None = None, profile: str | None = None) -> list[dict]:
        """Record spend against matching budgets. Returns list of breached budgets."""
        breached = []
        if cost <= 0:
            return breached

        with get_session(self._engine) as session:
            # Find matching active budgets
            query = session.query(Budget).filter(
                Budget.status.in_(["active", "exceeded"])
            )
            budgets = query.all()

            for b in budgets:
                # Check if budget matches the spend
                if b.key_id is not None and b.key_id != key_id:
                    continue
                if b.profile is not None and b.profile != profile:
                    continue

                # Record spend
                b.current_spend = (b.current_spend or 0.0) + cost
                spend_pct = (b.current_spend / b.amount * 100) if b.amount > 0 else 0

                # Check thresholds
                thresholds = [int(t.strip()) for t in b.threshold_pct.split(",") if t.strip().isdigit()]
                for t in sorted(thresholds):
                    if spend_pct >= t:
                        # Check if we already alerted for this threshold (avoid duplicates)
                        prev_pct = ((b.current_spend - cost) / b.amount * 100) if b.amount > 0 else 0
                        if prev_pct < t <= spend_pct:
                            breached.append({
                                "budget_id": b.id,
                                "budget_name": b.name,
                                "threshold": t,
                                "spend_pct": round(spend_pct, 1),
                                "current_spend": round(b.current_spend, 6),
                                "amount": b.amount,
                                "action": b.action,
                            })
                            b.last_alert_at = datetime.now(timezone.utc).isoformat()
                            logger.warning(
                                "budget_threshold_breached",
                                budget_id=b.id,
                                name=b.name,
                                threshold=t,
                                spend_pct=round(spend_pct, 1),
                            )

                # Check if exceeded
                if spend_pct >= 100 and b.status != "exceeded":
                    b.status = "exceeded"
                    logger.warning(
                        "budget_exceeded",
                        budget_id=b.id,
                        name=b.name,
                        spend_pct=round(spend_pct, 1),
                    )

            session.commit()

        return breached

    def check_spend_allowed(self, cost: float, key_id: int | None = None, profile: str | None = None) -> tuple[bool, str]:
        """Check if a spend is allowed. Returns (allowed, reason)."""
        if cost <= 0:
            return True, ""

        with get_session(self._engine) as session:
            query = session.query(Budget).filter(
                Budget.action == "block",
                Budget.status == "exceeded",
            )
            budgets = query.all()

            for b in budgets:
                if b.key_id is not None and b.key_id != key_id:
                    continue
                if b.profile is not None and b.profile != profile:
                    continue
                return False, f"budget exceeded: {b.name}"

        return True, ""


# ── Module-level singleton ────────────────────────────────────────────────
_budget_manager: BudgetManager | None = None


def get_budget_manager(engine=None) -> BudgetManager:
    """Get or create the budget manager singleton."""
    global _budget_manager
    if _budget_manager is None and engine is not None:
        _budget_manager = BudgetManager(engine)
    return _budget_manager


def init_budget_manager(engine) -> BudgetManager:
    """Force-initialize the budget manager."""
    global _budget_manager
    _budget_manager = BudgetManager(engine)
    return _budget_manager
