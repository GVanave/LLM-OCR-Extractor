"""Normalizers (plan Step 3).

They turn the many ways a value can be written ("16,500", "Rp. 16.500", "@16500")
into one canonical form, so the eval compares meaning and not formatting.

Locale assumption (CORD = Indonesian receipts): a "." or "," followed by exactly
3 digits is a thousands separator; followed by 1-2 digits it is a decimal point.
"""
import re

_NUM_CHARS = re.compile(r"[^0-9.,]")
_FIRST_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def parse_number(value) -> float | None:
    """'16,500' -> 16500.0, '23.000' -> 23000.0, '12.50' -> 12.5, '-9,545' -> -9545.0."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        for v in value:
            n = parse_number(v)
            if n is not None:
                return n
        return None

    s = str(value).strip()
    negative = s.startswith("-") or "(-" in s or (s.startswith("(") and s.endswith(")"))
    digits = _NUM_CHARS.sub("", s).strip(".,")
    if not any(c.isdigit() for c in digits):
        return None

    last_sep = max(digits.rfind("."), digits.rfind(","))
    if last_sep == -1:
        number = float(digits)
    else:
        int_part = re.sub(r"[.,]", "", digits[:last_sep]) or "0"
        frac = digits[last_sep + 1:]
        if len(frac) == 3:  # thousands separator
            number = float(int_part + frac)
        else:  # decimal point
            number = float(f"{int_part}.{frac}")
    return -number if negative else number


def parse_amount(value) -> float | None:
    """Money amount where the sign carries no meaning (discounts printed as '-9,545')."""
    n = parse_number(value)
    return abs(n) if n is not None else None


def parse_qty(value) -> float | None:
    """'3X' -> 3.0, '4.00xITEMs' -> 4.0, 'x2' -> 2.0, '1.00' -> 1.0."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _FIRST_NUMBER.search(str(value))
    if not m:
        return None
    return float(m.group().replace(",", "."))


def norm_name(value) -> str:
    """Uppercase, strip punctuation, collapse whitespace: 'Es  Teh-Manis.' -> 'ES TEH MANIS'."""
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    s = re.sub(r"[^\w%]+", " ", str(value).upper())
    return re.sub(r"\s+", " ", s).strip()


def numbers_equal(a: float | None, b: float | None, tol: float = 0.01) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol
