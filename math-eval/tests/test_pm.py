import pytest
import sys

from math_eval.main import parse_and_verify

@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # expansion
        (r"$\pm1$", r"$\pm 1$", True),
        (r"$\pm a$", r"$    \pm  a$", True),
        (r"$\pm{x}$", r"$    \pm  x$", True),
        (r"$\pm{x}$", r"$    \mp  x$", True),
        (r"これは$\pmx$です", r"これは$\pm x$です", True),
        (r"$3 \pm 1$", r"$3 \mp 1$", True),
        (r"$3 \pm 1$", r"$2, 4$", True),
        (r"$1 \pm \sqrt{2}$", r"$1+\sqrt{2}, 1-\sqrt{2}$", True),
        (r"x = 3 \pm 1", r"3 \mp 1", True),
        (r"$x = 3 \pm 1$", r"$3 \pm 1$", True),
        (r"$a+1\pm 1$", r"$a+2, a$", True),
        (r"$x = 3 \pm 1$", r"$3 \pm 1$", True),
        (r"${1 \pm \sqrt{97}} / 12$", r"${1\pm\sqrt{97}} / 12$", True),
        (r"${1 \pm \sqrt{97}}/ 12$", r"${1\pm\sqrt{97}} / 12$", True),
        (r"${1 \pm \sqrt{97}}/12$", r"${1\pm \sqrt{97}}/12$", True),
        (r"$\frac{1 \pm \sqrt{97}}{12}$", r"$\frac{1 \mp \sqrt{97}}{12}$", True),
        # Multiple \pm cases - 2^2 = 4 combinations
        (r"$1 \pm 2 \pm 3$", r"$6, 0, 2, -4$", True),  # 1+2+3=6, 1+2-3=0, 1-2+3=2, 1-2-3=-4
        (r"$1 \pm 2 \pm 3$", r"$1 \pm 2 \pm 3$", True),
        # Three \pm - 2^3 = 8 combinations
        (r"$1 \pm 1 \pm 1 \pm 1$", r"$4, 2, 2, 0, 2, 0, 0, -2$", True),
    ],
)

def test_verify(prediction: str, gold: str, expected: bool) -> None:
    result2 = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result2 == expected


@pytest.mark.parametrize(
    "prediction, gold, expected_original, expected_extended",
    [
        # All verify_cases from debug_pm_parse.py
        # Basic \pm with spacing differences
        (r"$\pm1$", r"$\pm 1$", True, True),
        # \pm expansion to explicit values
        (r"$3 \pm 1$", r"$2, 4$", True, True),
        # \pm with sqrt expansion
        (r"$1 \pm \sqrt{2}$", r"$1+\sqrt{2}, 1-\sqrt{2}$", True, True),
        # Without $ delimiter - original parse fails to expand properly
        (r"3 \pm 1", r"3 \mp 1", True, True),
        # \frac case - original parse returns string, _extended_parse expands to FiniteSet
        (r"$\frac{1 \pm \sqrt{97}}{12}$", r"$\frac{1 \mp \sqrt{97}}{12}$", False, True),
        # Multiple \pm - generates 4 combinations (2^2)
        # Original parse fails with multiple \pm, _extended_parse handles it correctly
        (r"$1 \pm 2 \pm 3$", r"$6, 0, 2, -4$", False, True),
        # Three \pm - 2^3 = 8 combinations
        (r"$1 \pm 1 \pm 1 \pm 1$", r"$4, 2, 2, 0, 2, 0, 0, -2$", False, True),
    ],
)
def test_extended_parse_improvement(
    prediction: str, gold: str, expected_original: bool, expected_extended: bool
) -> None:
    """Test cases comparing original parse vs _extended_parse behavior."""
    from math_verify import parse, verify
    from math_eval.main import _extended_parse

    # Test with original parse
    result_original = verify(parse(gold), parse(prediction))
    assert result_original == expected_original, f"Original parse: expected {expected_original}, got {result_original}"

    # Test with _extended_parse
    result_extended = verify(_extended_parse(gold), _extended_parse(prediction))
    assert result_extended == expected_extended, f"Extended parse: expected {expected_extended}, got {result_extended}"
