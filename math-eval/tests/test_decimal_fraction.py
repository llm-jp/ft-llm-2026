r"""test_decimal_fraction.py

小数を分数に変換して再比較するフォールバックのテスト。
"""

import pytest

from math_eval.main import _decimal_to_fraction, parse_and_verify


# --- _decimal_to_fraction 単体テスト ---


class TestDecimalToFraction:
    r"""_decimal_to_fraction の単体テスト。"""

    def test_simple_decimal(self) -> None:
        assert _decimal_to_fraction("0.5") == r"\frac{5}{10}"

    def test_decimal_625(self) -> None:
        assert _decimal_to_fraction("0.625") == r"\frac{625}{1000}"

    def test_decimal_with_variable(self) -> None:
        assert _decimal_to_fraction("0.625x") == r"\frac{625}{1000}x"

    def test_no_decimal(self) -> None:
        assert _decimal_to_fraction("x + 1") == "x + 1"


# --- 統合テスト ---


@pytest.mark.parametrize(
    "prediction, gold, method, expected",
    [
        # 0.625x と \frac{5}{8}x は等しい
        (r"$0.625x$", r"$\frac{5}{8}x$", "soft", True),
        # \frac{5}{8}x と 0.625x（gold が小数）
        (r"$\frac{5}{8}x$", r"$0.625x$", "soft", True),
        # 0.5 と \frac{1}{2}
        (r"$0.5$", r"$\frac{1}{2}$", "soft", True),
        # 不一致: 0.5 と \frac{1}{3}
        (r"$0.5$", r"$\frac{1}{3}$", "soft", False),
        # 不一致: 0.625x と \frac{1}{8}x（誤答）
        (r"$0.625x$", r"$\frac{1}{8}x$", "soft", False),
    ],
)
def test_decimal_fraction_integration(
    prediction: str, gold: str, method: str, expected: bool
) -> None:
    """小数 → 分数変換フォールバックが parse_and_verify で正しく機能することを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method=method)
    assert result == expected
