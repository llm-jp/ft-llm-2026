"""test_delimiter_fallback.py

prediction にデリミタがない場合、$...$ で囲んでフォールバック評価することを確認する。
"""

import pytest

from math_eval.main import parse_and_verify


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # デリミタなしの LaTeX 式がフォールバックで正しく評価される
        (r"\sqrt{2}", r"$\sqrt{2}$", True),
        (r"\frac{1}{2}", r"$\frac{1}{2}$", True),
        (r"2x+1", r"$2x+1$", True),
        (r"x^2 + 3x - 5", r"$x^2 + 3x - 5$", True),
        # デリミタありの場合はそのまま評価（フォールバック不要）
        (r"$\sqrt{2}$", r"$\sqrt{2}$", True),
        (r"\boxed{42}", r"$42$", True),
        # 数値はフォールバックなしでも正しく評価される
        (r"3", r"$3$", True),
        (r"3.14", r"$3.14$", True),
        # 不一致はフォールバックしても不一致
        (r"\sqrt{3}", r"$\sqrt{2}$", False),
        (r"2x+1", r"$3x+1$", False),
    ],
)
def test_delimiter_fallback(prediction: str, gold: str, expected: bool) -> None:
    """prediction にデリミタがない場合のフォールバック評価を確認する。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected
