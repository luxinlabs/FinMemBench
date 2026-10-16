from .models import Evidence, Query, Decision

def decide(events: list[Evidence], query: Query) -> Decision:
    """Simple as-of, latest-event baseline. Not a learned policy.

    Events must represent adjudicated updates to the SAME claim key;
    do not use on raw XBRL values without context normalization.
    """
    eligible = [e for e in events if e.claim_key == query.claim_key and e.available_at <= query.as_of]
    if not eligible:
        return Decision("abstain", None, None, "no admissible evidence")
    eligible.sort(key=lambda e: (e.available_at, e.accession))
    latest_time = eligible[-1].available_at
    tied = [e for e in eligible if e.available_at == latest_time]
    # Accession order is not a temporal order: evidence that became available at
    # the same instant with different values or states is an unresolved conflict.
    if len({(e.status, e.value) for e in tied}) > 1:
        return Decision("abstain", None, tied[0].accession, "conflicting simultaneous evidence")
    latest = tied[0]
    if latest.status == "non_reliance":
        return Decision("abstain", None, latest.accession, "withdrawn evidence")
    if latest.status not in ("valid", "supersede"):
        return Decision("abstain", None, latest.accession, "unknown evidence state")
    return Decision("answer", latest.value, latest.accession, "latest admissible evidence")
