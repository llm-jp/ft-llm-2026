import pytest

from math_eval.main import parse_and_verify


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # Numerical extraction: Positive integers
        (r"1", r"1", True),
        (r"42", r"42", True),
        (r"1", r"2", False),
        (r"100", r"1000", False),
        # Numerical extraction: Negative integers
        (r"-1", r"-1", True),
        (r"-42", r"-42", True),
        (r"-1", r"-2", False),
        (r"-100", r"100", False),
        # Numerical extraction: Positive decimals
        (r"3.14", r"3.14", True),
        (r"0.001", r"0.001", True),
        (r".5", r"0.5", True),
        (r"2.71", r"2.72", False),
        (r"1.0", r"1.1", False),
        (r".5", r"0.6", False),
        # Numerical extraction: Negative decimals
        (r"-0.001", r"-0.001", True),
        (r"-0.1", r"-0.1", True),
        (r"-0,5", r"-0.7", False),
        (r"-2.71", r"-2.70", False),
        # Numerical extraction: Integer vs Decimal
        (r"5.0", r"5", True),
        (r"10.0", r"11", False),
        # Thousands separator
        (r"1,000", r"1000", True),
        (r"20,500", r"2,500", False),
        # Numerical extraction: Units
        (r"5 kg", r"5 kilograms", True),
        (r"3 m", r"3 meters", True),
        (r"$100", r"100 dollars", True),
        (r"10 kg", r"12 kilograms", False),
        (r"5 m", r"6 meters", False),
        (r"1 m", r"100 cm", False),
    ],
)
def test_verify(prediction: str, gold: str, expected: bool) -> None:
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected
