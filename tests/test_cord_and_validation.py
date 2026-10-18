"""Real CORD labels (validation split) converted and checked."""
from evals.cord import convert_gt
from receipt.validation import check_receipt, is_consistent

SINGLE_ITEM_WITH_SUBS = {
    "menu": {"nm": "M-Caramel Black Tea", "unitprice": "@28,000", "cnt": "1X", "price": "28,000",
             "sub": [{"nm": "70%"}, {"nm": "Less Ice"}]},
    "sub_total": {"subtotal_price": "28,000", "tax_price": "0"},
    "total": {"total_price": "28,000", "cashprice": "28,000", "changeprice": "0"},
}
TAX_INCLUSIVE = {
    "menu": {"nm": "LE MINERAL", "cnt": "1.00", "price": "8,000"},
    "sub_total": {"subtotal_price": "7,273", "tax_price": "727"},
    "total": {"total_price": "8,000", "cashprice": "8,000"},
}
RECEIPT_DISCOUNT = {
    "menu": {"nm": "Choco Devil", "cnt": "4", "discountprice": "-9,545", "price": "63,636"},
    "sub_total": {"subtotal_price": "63,636", "discount_price": "-9,545", "tax_price": "5,409"},
    "total": {"total_price": "59,500", "cashprice": "100,000", "changeprice": "40,500"},
}
ITEM_DISCOUNT = {
    "menu": [{"nm": "TALAM UNGU", "unitprice": "@6500", "cnt": "3X", "discountprice": "-7,800", "price": "19,500"},
             {"nm": "MIKA KECIL", "unitprice": "@0", "cnt": "1X", "price": "0"}],
    "sub_total": {"subtotal_price": "11,700"},
    "total": {"total_price": "11,700", "cashprice": "20,000", "changeprice": "8,300", "menuqty_cnt": "4.00xITEMs"},
}


PRICED_ADD_ON = {  # dev_0071: the add-on price counts toward the subtotal
    "menu": {"nm": "Dum Dum Thai Coffee", "cnt": "1", "price": "19.000", "sub": {"nm": "Ice", "cnt": "1", "price": "1.000"}},
    "sub_total": {"subtotal_price": "20.000"},
    "total": {"total_price": "20.000", "cashprice": "50.000", "changeprice": "30.000"},
}
NEGATIVE_CHANGE = {  # dev_0013: change printed as "-4,500"
    "menu": [{"nm": "MINERAL 600 ML", "cnt": "1", "price": "7,727"}, {"nm": "BULGOGI RICE R", "cnt": "1", "price": "33,636"}],
    "sub_total": {"subtotal_price": "41,364", "tax_price": "4,136"},
    "total": {"total_price": "45,500", "cashprice": "50,000", "changeprice": "-4,500"},
}


def test_priced_add_on_and_negative_change_are_consistent():
    assert is_consistent(convert_gt(PRICED_ADD_ON))
    r = convert_gt(NEGATIVE_CHANGE)
    assert r["totals"]["change"] == 4500 and is_consistent(r)


def test_convert_single_item_dict_with_sub_items():
    r = convert_gt(SINGLE_ITEM_WITH_SUBS)
    assert len(r["items"]) == 1
    item = r["items"][0]
    assert item == {
        "name": "M-Caramel Black Tea", "qty": 1, "unit_price": 28000, "price": 28000, "discount": None,
        "sub_items": [{"name": "70%", "qty": None, "price": None}, {"name": "Less Ice", "qty": None, "price": None}],
        "box_2d": None,
    }
    assert r["totals"]["tax"] == 0 and r["totals"]["change"] == 0 and r["totals"]["service"] is None


def test_convert_discounts_are_positive():
    r = convert_gt(ITEM_DISCOUNT)
    assert r["items"][0]["discount"] == 7800
    assert convert_gt(RECEIPT_DISCOUNT)["totals"]["discount"] == 9545


def test_convert_list_name_and_missing_sections():
    r = convert_gt({"menu": [{"nm": ["Nasi", "Goreng"], "price": "20.000"}], "total": {"total_price": "20.000"}})
    assert r["items"][0]["name"] == "Nasi Goreng"
    assert r["totals"]["subtotal"] is None and r["totals"]["total"] == 20000


def test_real_labels_are_consistent():
    for gt in (SINGLE_ITEM_WITH_SUBS, TAX_INCLUSIVE, RECEIPT_DISCOUNT, ITEM_DISCOUNT):
        r = convert_gt(gt)
        assert is_consistent(r), [c for c in check_receipt(r) if not c.passed]


def test_wrong_total_fails():
    r = convert_gt(RECEIPT_DISCOUNT)
    r["totals"]["total"] = 69500
    failed = [c.name for c in check_receipt(r) if not c.passed]
    assert "subtotal -> total" in failed and "cash - total = change" in failed


def test_missed_item_fails_items_check():
    r = convert_gt(ITEM_DISCOUNT)
    r["items"].pop(0)
    assert "items = subtotal" in [c.name for c in check_receipt(r) if not c.passed]


def test_no_checks_apply():
    assert is_consistent({"items": [{"name": "x", "price": None}], "totals": {}}) is None
