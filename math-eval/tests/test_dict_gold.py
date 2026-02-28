r"""test_dict_gold.py

Dict 形式の prediction ({key: value, ...}) に対する正規化・判定テスト。
- _parse_dict_notation の単体テスト
- parse_and_verify を通じた統合テスト
"""

import pytest

from math_eval.main import _parse_dict_notation, parse_and_verify


# --- _parse_dict_notation 単体テスト ---


class TestParseDictNotation:
    """_parse_dict_notation の単体テスト。"""

    def test_boxed_numeric_keys(self) -> None:
        result = _parse_dict_notation(r"\boxed{{0: 1/6, 1: 1/3, 2: 1/2}}")
        assert result == ["$1/6, 1/3, 1/2$"]

    def test_variable_keys(self) -> None:
        result = _parse_dict_notation(r"${x: 3, y: 7}$")
        assert result is not None
        assert result == ["$x = 3, y = 7$"]

    def test_variable_keys_latex_delim(self) -> None:
        result = _parse_dict_notation(r"$\{a: 5, b: 11\}$")
        assert result == ["$a = 5, b = 11$"]

    def test_latex_frac_values(self) -> None:
        result = _parse_dict_notation(r"\boxed{{0: \frac{1}{6}, 1: \frac{5}{6}}}")
        assert result == [r"$\frac{1}{6}, \frac{5}{6}$"]

    def test_not_dict(self) -> None:
        assert _parse_dict_notation(r"$x + 1$") is None

    def test_no_colon(self) -> None:
        assert _parse_dict_notation(r"{1, 2, 3}") is None


# --- 統合テスト ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # 数値キーの dict pred vs 値リスト gold
        (
            r"\boxed{{0: \frac{1}{6}, 1: \frac{1}{3}, 2: \frac{1}{2}}}",
            r"$\frac{1}{6}, \frac{1}{3}, \frac{1}{2}$",
            True,
        ),
        # 数値キーの dict pred vs P(X=k) 形式 gold
        (
            r"\boxed{{0: \frac{1}{6}, 1: \frac{1}{3}, 2: \frac{1}{2}}}",
            r"$P(X=0)=\frac{1}{6},\; P(X=1)=\frac{1}{3},\; P(X=2)=\frac{1}{2}$",
            True,
        ),
        # 変数キーの dict pred vs 方程式 gold
        (
            r"\boxed{{x: 3, y: 7}}",
            r"$x = 3, y = 7$",
            True,
        ),
        # pred が dict で値が不一致
        (
            r"\boxed{{x: 3, y: 7}}",
            r"$x = 3, y = 11$",
            False,
        ),
        # 部分一致は不正解
        (
            r"\boxed{{0: \frac{1}{6}}}",
            r"$\frac{1}{6}, \frac{1}{3}, \frac{1}{2}$",
            False,
        ),
        # 中括弧なし: \boxed{x: 3, y: 7}
        (
            r"\boxed{x: 3, y: 7}",
            r"$x = 3, y = 7$",
            True,
        ),
        # 中括弧なし: 裸の dict 風テキスト
        (
            r"0: 1/6, 1: 1/3, 2: 1/2",
            r"$\frac{1}{6}, \frac{1}{3}, \frac{1}{2}$",
            True,
        ),
    ],
)
def test_dict_integration(
    prediction: str, gold: str, expected: bool
) -> None:
    """Dict 形式のフォールバックが parse_and_verify で正しく機能することを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected
