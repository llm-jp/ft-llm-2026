import pytest

from math_eval.main import parse_and_verify
from math_eval.main import EvaluationMethod

@pytest.mark.parametrize(
    "prediction, gold, evaluation_method, expected",
    [
        (r"$x+1$", r"$1+x$", "soft", True),
        (r"$x^2$", r"$x * x$", "strict", False),
        (r"$i^2$", r"$-1$", "complex", True),
    ],
)
def test_verify(prediction: str, gold: str, evaluation_method: EvaluationMethod, expected: bool) -> None:
    result = parse_and_verify(prediction, gold, evaluation_method)
    assert result == expected

def test_invalid_evaluation_method() -> None:
    """Test that invalid evaluation method raises ValueError."""
    with pytest.raises(ValueError, match="Unknown evaluation method"):
        parse_and_verify("1", "1", evaluation_method="hard")  # type: ignore

def test_none_evaluation_method_warning() -> None:
    """Test that None evaluation_method raises a warning."""
    with pytest.warns(UserWarning): 
        result = parse_and_verify("1", "1", evaluation_method=None)
        assert result is True

def test_empty_evaluation_method_warning() -> None:
    """Test that None evaluation_method raises a warning."""
    with pytest.warns(UserWarning): 
        result = parse_and_verify("1", "1")
        assert result is True

