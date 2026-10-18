"""Math consistency checks (plan Steps 4 and 8).

Each check only runs when the values it needs are present. A receipt is
"consistent" when every check that applies passes.

Receipts are messy, so each check accepts the common real-world variants:
- item prices may or may not include tax (items = subtotal OR items = subtotal + tax)
- discounts may sit on items or on the whole receipt
"""
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _close(a: float, b: float) -> bool:
    """Allow rounding differences: 1 unit or 1%, whichever is larger."""
    return abs(a - b) <= max(1.0, 0.01 * abs(b))


def _v(d: dict, key: str) -> float | None:
    v = d.get(key)
    return float(v) if v is not None else None


def check_receipt(receipt: dict) -> list[Check]:
    items = receipt.get("items") or []
    t = receipt.get("totals") or {}
    subtotal, tax, service = _v(t, "subtotal"), _v(t, "tax"), _v(t, "service")
    discount, total = _v(t, "discount"), _v(t, "total")
    cash, change = _v(t, "cash"), _v(t, "change")
    checks: list[Check] = []

    # 1. qty x unit_price = line price
    for i, it in enumerate(items):
        qty, unit, price = _v(it, "qty"), _v(it, "unit_price"), _v(it, "price")
        if qty and unit and price is not None:
            checks.append(Check(
                f"item[{i}] qty*unit_price", _close(qty * unit, price),
                f"{qty:g} x {unit:g} = {qty * unit:g}, printed {price:g}",
            ))

    # 2. items add up to subtotal (or to total when there is no subtotal)
    prices = [_v(it, "price") for it in items]
    if items and all(p is not None for p in prices):
        add_ons = sum(_v(s, "price") or 0 for it in items for s in it.get("sub_items") or [])
        gross = sum(prices) + add_ons  # priced add-ons ("+ Ice 1.000") count toward the subtotal
        net = gross - sum(_v(it, "discount") or 0 for it in items)
        target_name, target = ("subtotal", subtotal) if subtotal is not None else ("total", total)
        if target is not None:
            targets = [target]
            if target_name == "subtotal" and tax is not None:  # tax-inclusive item prices
                targets.append(target + tax)
            ok = any(_close(c, x) for c in (gross, net) for x in targets)
            checks.append(Check(
                f"items = {target_name}", ok,
                f"items gross {gross:g} / net {net:g}, {target_name} {target:g}",
            ))

    # 3. subtotal + tax + service - discount = total
    if subtotal is not None and total is not None:
        full = subtotal + (tax or 0) + (service or 0) - (discount or 0)
        ok = _close(full, total) or _close(subtotal, total) or _close(subtotal - (discount or 0), total)
        checks.append(Check(
            "subtotal -> total", ok,
            f"{subtotal:g} + tax {tax or 0:g} + service {service or 0:g} - discount {discount or 0:g} "
            f"= {full:g}, total {total:g}",
        ))

    # 4. cash - total = change
    if cash and total is not None and change is not None:
        change = abs(change)  # some receipts print change as "-4,500"
        checks.append(Check(
            "cash - total = change", _close(cash - total, change),
            f"{cash:g} - {total:g} = {cash - total:g}, change {change:g}",
        ))

    return checks


def is_consistent(receipt: dict) -> bool | None:
    """True/False, or None when no check applies."""
    checks = check_receipt(receipt)
    if not checks:
        return None
    return all(c.passed for c in checks)
