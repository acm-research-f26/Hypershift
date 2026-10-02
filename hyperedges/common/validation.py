"""Internal identifier and scalar checks shared by independent constructors."""

from numbers import Integral

__all__ = []


def _validate_identifiers(values, kind, *, allow_empty=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{kind} identifiers must be a sequence")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{kind} identifiers must be a sequence") from exc
    if (not values and not allow_empty) or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"{kind} identifiers must be nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {kind} identifiers")
    return values



def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")

