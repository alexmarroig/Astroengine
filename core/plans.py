import time
from dataclasses import dataclass

TRIAL_SECONDS = 7 * 24 * 60 * 60

@dataclass
class UserPlan:
    user_id: str
    plan: str  # "free" | "trial" | "premium"
    trial_started_at: float

# memória (MVP). Depois a gente pluga em Supabase/DB.
_users: dict[str, UserPlan] = {}
_premium_users: set[str] = set()  # se quiser marcar premium manualmente

def get_user_plan(user_id: str) -> UserPlan:
    now = time.time()

    if user_id in _users:
        u = _users[user_id]
    else:
        u = UserPlan(user_id=user_id, plan="trial", trial_started_at=now)
        _users[user_id] = u

    if user_id in _premium_users:
        u.plan = "premium"
        return u

    # expira trial -> free
    if u.plan == "trial" and (now - u.trial_started_at) > TRIAL_SECONDS:
        u.plan = "free"

    return u

def is_trial_or_premium(plan: str) -> bool:
    return plan in ("trial", "premium")
