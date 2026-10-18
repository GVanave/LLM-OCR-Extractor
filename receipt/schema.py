"""Target schema for receipt extraction (plan Step 2).

Bump SCHEMA_VERSION whenever a field is added, removed or changes meaning,
so old eval runs and cached responses are never mixed with new ones.
"""
from pydantic import BaseModel

SCHEMA_VERSION = "receipt-v1"


class SubItem(BaseModel):
    name: str
    qty: float | None
    price: float | None


class Item(BaseModel):
    name: str
    qty: float | None
    unit_price: float | None
    price: float | None  # line total for this item
    discount: float | None  # item-level discount, positive number
    sub_items: list[SubItem]
    box_2d: list[int] | None  # [ymin, xmin, ymax, xmax], normalized 0-1000


class Totals(BaseModel):
    subtotal: float | None
    tax: float | None
    service: float | None
    discount: float | None  # receipt-level discount, positive number
    total: float | None
    cash: float | None
    change: float | None


class Receipt(BaseModel):
    items: list[Item]
    totals: Totals
    currency: str | None


TOTAL_FIELDS = list(Totals.model_fields)
