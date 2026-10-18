import pytest

from receipt.normalize import norm_name, numbers_equal, parse_amount, parse_number, parse_qty


@pytest.mark.parametrize("raw, expected", [
    ("16,500", 16500), ("23.000", 23000), ("1,234,567", 1234567), ("1.234.567", 1234567),
    ("12.50", 12.5), ("1.00", 1.0), ("1,5", 1.5), ("7273", 7273),
    ("-9,545", -9545), ("@6500", 6500), (":9,000", 9000), ("Rp. 12.500", 12500), ("Rp12,500.00", 12500),
    (16500, 16500), (12.5, 12.5), (["50.000"], 50000),
    (None, None), ("", None), ("abc", None), ("-", None),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_parse_amount_drops_sign():
    assert parse_amount("-9,545") == 9545
    assert parse_amount(None) is None


@pytest.mark.parametrize("raw, expected", [
    ("1", 1), ("3X", 3), ("x2", 2), ("4.00xITEMs", 4), ("1.00", 1), ("2 pcs", 2), (None, None), ("x", None),
])
def test_parse_qty(raw, expected):
    assert parse_qty(raw) == expected


def test_norm_name():
    assert norm_name("  Es  Teh-Manis. ") == "ES TEH MANIS"
    assert norm_name(["Kopi", "Susu"]) == "KOPI SUSU"
    assert norm_name("70%") == "70%"
    assert norm_name(None) == ""


def test_numbers_equal():
    assert numbers_equal(16500, 16500.0)
    assert numbers_equal(None, None)
    assert not numbers_equal(None, 0)
    assert not numbers_equal(16500, 16501)
