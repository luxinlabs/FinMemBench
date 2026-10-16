import math
from typing import Iterable, Optional

from .models import Decision


def values_equal(a: Optional[str], b: Optional[str], rel_tol: float = 1e-9) -> bool:
    """Numeric-aware equality: "90", "90.0" and "9e1" are the same reported value."""
    if a is None or b is None:
        return a is b
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=0.0)
    except ValueError:
        return a.strip() == b.strip()


def score(decision: Decision, expected: str | None,
          admissible_accessions: Optional[Iterable[str]] = None) -> dict:
    """Score one decision. `expected is None` means abstention is the reference action.

    If `admissible_accessions` is given, an answer is `supported` only when it cites
    one of them (evidence available at the cutoff that reports the expected value).
    """
    answered = decision.action == "answer"
    correct = answered and expected is not None and values_equal(decision.value, expected)
    if admissible_accessions is None:
        supported = correct
    else:
        supported = correct and decision.accession in set(admissible_accessions)
    return {"answered": answered, "correct": correct, "supported": supported,
            "appropriate_abstention": not answered and expected is None}
