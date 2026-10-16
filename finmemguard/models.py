from dataclasses import dataclass
from datetime import datetime
from typing import Optional

STATUSES = ("valid", "supersede", "non_reliance")
ACTIONS = ("answer", "abstain")


def _require_aware(ts: datetime, name: str) -> None:
    # Comparing naive and aware datetimes raises TypeError mid-run; naive
    # timestamps also make the as-of boundary ambiguous. Reject both early.
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got {ts!r}")


@dataclass(frozen=True)
class Evidence:
    accession: str
    claim_key: str
    value: str
    available_at: datetime
    status: str = "valid"  # valid, supersede, non_reliance

    def __post_init__(self):
        _require_aware(self.available_at, "available_at")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {STATUSES}")


@dataclass(frozen=True)
class Query:
    claim_key: str
    as_of: datetime

    def __post_init__(self):
        _require_aware(self.as_of, "as_of")


@dataclass(frozen=True)
class Decision:
    action: str  # answer or abstain
    value: Optional[str]
    accession: Optional[str]
    reason: str
