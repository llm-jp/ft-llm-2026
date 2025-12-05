import pytest

from math_eval.main import parse_and_verify

@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # expansion
        (r"$(x+1)^2$", r"$x^2+2x+1$", False), 
        (r"$1 + 2x + x^2$", r"$x^2+2x+1$", True), 

        # calculation
        (r"$\sqrt{2}^2$", r"$2$", False), 
        (r"$5a-8a$", r"$-3a$", False),
        (r"$1+2$", r"$3$", False),

        # division
        (r"$6/2$", r"$3/1$", True), 

        # different expression
        (r"$2\sqrt{2}$", r"$2^{3/2}$", False), 
    ],
)

def test_verify(prediction: str, gold: str, expected: bool) -> None:
    result = parse_and_verify(prediction, gold, evaluation_method="strict")
    assert result == expected


