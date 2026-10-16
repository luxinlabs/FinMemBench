from datetime import datetime, timezone
from .models import Evidence, Query
from .controller import decide

def utc(s: str):
    ts = datetime.fromisoformat(s)
    # Only naive strings are assumed UTC; an explicit offset must not be overwritten.
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)

def main():
    events = [Evidence("EXAMPLE-001", "demo:revenue:FY2024", "100", utc("2025-02-01T00:00:00")),
              Evidence("EXAMPLE-002", "demo:revenue:FY2024", "90", utc("2025-04-01T00:00:00"), "supersede")]
    for cutoff in ("2025-03-01T00:00:00", "2025-05-01T00:00:00"):
        print(cutoff, decide(events, Query("demo:revenue:FY2024", utc(cutoff))))
    print("SYNTHETIC ONLY — NOT SEC DATA OR RESEARCH RESULTS")

if __name__ == "__main__":
    main()
