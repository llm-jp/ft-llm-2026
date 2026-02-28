r"""test_numeric_approx.py

小数近似比較のテスト。
小数 vs 分数/無理数を、小数の桁数に合わせて丸めて比較する。
soft・strict 両方で動作する。
"""

import pytest

from math_eval.main import _count_decimal_places, parse_and_verify


# --- _count_decimal_places 単体テスト ---


class TestCountDecimalPlaces:
    """_count_decimal_places の単体テスト。"""

    def test_three_places(self) -> None:
        assert _count_decimal_places("0.333") == 3

    def test_four_places(self) -> None:
        assert _count_decimal_places("3.1416") == 4

    def test_one_place(self) -> None:
        assert _count_decimal_places("3.0") == 1

    def test_no_decimal(self) -> None:
        assert _count_decimal_places("42") is None

    def test_negative(self) -> None:
        assert _count_decimal_places("-0.667") == 3

    def test_multiple_decimals_returns_min(self) -> None:
        assert _count_decimal_places("0.33, 0.6667") == 2

    def test_latex_no_decimal(self) -> None:
        assert _count_decimal_places(r"\frac{1}{3}") is None


# --- 統合テスト ---


@pytest.mark.parametrize(
    "prediction, gold, evaluation_method, expected",
    [
        # --- soft ---
        # 分数 vs 小数 (桁数一致)
        ("0.333", r"$\frac{1}{3}$", "soft", True),
        ("0.6667", r"$\frac{2}{3}$", "soft", True),
        # 分数 vs 小数 (桁数不一致)
        ("0.334", r"$\frac{1}{3}$", "soft", False),
        ("0.3334", r"$\frac{1}{3}$", "soft", False),
        # 無理数 vs 小数
        ("3.14", r"$\pi$", "soft", True),
        ("3.1416", r"$\pi$", "soft", True),
        ("3.15", r"$\pi$", "soft", False),
        ("1.414", r"$\sqrt{2}$", "soft", True),
        ("1.4142", r"$\sqrt{2}$", "soft", True),
        # --- strict ---
        ("0.333", r"$\frac{1}{3}$", "strict", True),
        ("0.334", r"$\frac{1}{3}$", "strict", False),
        ("3.14", r"$\pi$", "strict", True),
        ("1.414", r"$\sqrt{2}$", "strict", True),
    ],
)
def test_numeric_approx(
    prediction: str, gold: str, evaluation_method: str, expected: bool
) -> None:
    """小数近似比較が parse_and_verify で正しく機能することを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method=evaluation_method)
    assert result == expected
