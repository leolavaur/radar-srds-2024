"""Pytest tests for the utils module."""

from .utils import load_json_metrics, zipd


def test_json_load_metrics():
    """Test that JSON-serialized values are deserialized."""
    dct = {
        "a": 1.0,
        "b": "2",
        "c": {"c1": 3, "c2": '{"abc":"1"}'},
    }
    output = load_json_metrics(dct)
    assert output == {
        "a": 1.0,
        "b": 2,
        "c": {"c1": 3, "c2": {"abc": "1"}},
    }


def test_zipd():
    """Test that zipd-ing two dicts makes a list of tuples from their common keys."""
    dct1 = {"a": 1, "b": 2, "c": 3}
    dct2 = {"a": 4, "b": 5, "c": 6}
    dct3 = {"c": 7, "d": 8, "e": 9}
    dct4 = {"f": 10, "g": 11, "h": 12}
    assert list(zipd(dct1)) == [("a", 1), ("b", 2), ("c", 3)]
    assert list(zipd(dct1, dct2)) == [("a", 1, 4), ("b", 2, 5), ("c", 3, 6)]
    assert list(zipd(dct1, dct2, dct3)) == [("c", 3, 6, 7)]
    assert list(zipd(dct3, dct1, dct2)) == [("c", 7, 3, 6)]
    assert list(zipd(dct1, dct4)) == []
