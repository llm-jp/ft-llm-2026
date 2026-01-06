import pytest

from math_eval.main import parse_and_verify


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        (r"$1/i$", r"$-i$", True),
        (r"$e^{i\theta} + e^{-i\theta}$", r"2\cos\theta", True),
        (r"$e^{ix} / (\cos x + i \sin x)$", "1", True),
        (r"$e^{ix} * (\cos x - i \sin x)$", "1", True),
        (r"$e^{ix}$", r"$1/e^{-ix}$", True),
        (r"$e^{ix}$", r"$\cos x + i \sin x$", True),
        (r"$e^{i\pi/2}$", r"$i$", True),
    ],
)
def test_verify(prediction: str, gold: str, expected: bool) -> None:
    result = parse_and_verify(prediction, gold, evaluation_method="complex")
    assert result == expected
