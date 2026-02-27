"""test_pm.py

\\pm / \\mp 展開のテスト。
- 基本的な展開と verify
- Equality (x = ...) 内の \\pm 展開と FiniteSet 統合
- _extended_parse と parse の比較
"""

import pytest

from math_eval.main import parse_and_verify, _extended_parse
from sympy import Eq, FiniteSet, srepr, S


# --- 基本的な \\pm 展開テスト ---


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


# --- _extended_parse と parse の比較テスト ---


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

    # Test with original parse
    result_original = verify(parse(gold), parse(prediction))
    assert result_original == expected_original, f"Original parse: expected {expected_original}, got {result_original}"

    # Test with _extended_parse
    result_extended = verify(_extended_parse(gold), _extended_parse(prediction))
    assert result_extended == expected_extended, f"Extended parse: expected {expected_extended}, got {result_extended}"


# --- Equality + \pm の FiniteSet 統合テスト ---


class TestEqualityPmMerge:
    r"""x = a \pm b 形式の式で、Equality 内の FiniteSet が正しく統合されることを確認する。"""

    def test_eq_pm_simple(self):
        r"""$x = 3 \pm 1$ → Eq(x, {2, 4}) に統合される。"""
        result = _extended_parse(r"$x = 3 \pm 1$")
        assert len(result) == 1
        val = result[0]
        assert isinstance(val, Eq), f"Expected Eq, got {type(val).__name__}"
        rhs = val.rhs
        assert isinstance(rhs, FiniteSet), f"Expected FiniteSet, got {type(rhs).__name__}"
        assert set(rhs) == {S(2), S(4)}

    def test_eq_pm_sqrt(self):
        r"""$x = -3 \pm \sqrt{6}$ → Eq(x, {-3-sqrt(6), -3+sqrt(6)}) に統合される。"""
        result = _extended_parse(r"$x = -3 \pm \sqrt{6}$")
        assert len(result) == 1
        val = result[0]
        assert isinstance(val, Eq)
        rhs = val.rhs
        assert isinstance(rhs, FiniteSet)
        assert len(rhs) == 2

    def test_eq_pm_with_comma(self):
        r"""$x = 1, 3 \pm 4$ → Eq(x, {-1, 1, 7}) に統合（入れ子にならない）。"""
        result = _extended_parse(r"$x = 1, 3 \pm 4$")
        assert len(result) == 1
        val = result[0]
        assert isinstance(val, Eq), f"Expected Eq, got {type(val).__name__}"
        rhs = val.rhs
        assert isinstance(rhs, FiniteSet), f"Expected FiniteSet, got {type(rhs).__name__}"
        # 入れ子にならず、全要素がフラットに統合されている
        for elem in rhs:
            assert not isinstance(elem, (FiniteSet, Eq)), \
                f"FiniteSet element should not be nested: {elem} ({type(elem).__name__})"
        assert set(rhs) == {S(-1), S(1), S(7)}

    def test_eq_pm_multiple(self):
        r"""$x = 1 \pm 2 \pm 3$ → Eq(x, {-4, 0, 2, 6}) に統合。"""
        result = _extended_parse(r"$x = 1 \pm 2 \pm 3$")
        assert len(result) == 1
        val = result[0]
        assert isinstance(val, Eq)
        rhs = val.rhs
        assert isinstance(rhs, FiniteSet)
        assert set(rhs) == {S(-4), S(0), S(2), S(6)}

    def test_eq_pm_no_nested_equality(self):
        r"""Equality が FiniteSet の要素に入れ子にならないことを確認する。"""
        result = _extended_parse(r"$x = 1, 3 \pm 4$")
        val = result[0]
        # FiniteSet(Eq(...), Eq(...)) のような入れ子構造ではないことを確認
        assert isinstance(val, Eq), \
            f"Result should be Eq, not {type(val).__name__}: {val}"
        assert not isinstance(val, FiniteSet) or not any(isinstance(e, Eq) for e in val), \
            f"FiniteSet should not contain Equality objects: {val}"

    def test_eq_pm_lhs_preserved(self):
        r"""統合後も LHS (変数名) が保持される。"""
        result = _extended_parse(r"$y = 5 \pm 2$")
        val = result[0]
        assert isinstance(val, Eq)
        assert str(val.lhs) == "y"
        assert set(val.rhs) == {S(3), S(7)}

    def test_eq_pm_decimal(self):
        r"""小数を含む Equality + \pm。"""
        result = _extended_parse(r"$x = 1.5 \pm 0.5$")
        val = result[0]
        assert isinstance(val, Eq)
        rhs = val.rhs
        assert isinstance(rhs, FiniteSet)
        assert len(rhs) == 2


# --- Equality + \pm の parse_and_verify 統合テスト ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # Eq + \pm が gold のカンマ区切りと一致
        (r"$x = 3 \pm 1$", r"$x = 2, 4$", True),
        # 逆順: gold が \pm、pred がカンマ区切り
        (r"$x = 2, 4$", r"$x = 3 \pm 1$", True),
        # カンマ付き + \pm
        (r"$x = 1, 3 \pm 4$", r"$x = -1, 1, 7$", True),
        # 値が異なる場合は不一致
        (r"$x = 3 \pm 1$", r"$x = 1, 5$", False),
        # \pm なしの Equality（回帰テスト）
        (r"$x = 5$", r"$x = 5$", True),
        (r"$x = 5$", r"$x = 6$", False),
    ],
)
def test_eq_pm_verify(prediction: str, gold: str, expected: bool) -> None:
    r"""Equality 内の \pm が parse_and_verify で正しく判定されることを確認する。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected


# --- FiniteSet (非 Equality) の \pm 統合テスト（回帰テスト） ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # 基本的な FiniteSet + \pm
        (r"$1, 3 \pm 4$", r"$\{-1, 1, 7\}$", True),
        (r"$1.2, 3 \pm 4$", r"$-1, 1.2, 7$", True),
        # \pm 同士の比較
        (r"$3 \pm 1$", r"$3 \mp 1$", True),
        # 値が異なる
        (r"$1, 3 \pm 4$", r"$\{-1, 1, 8\}$", False),
    ],
)
def test_finiteset_pm_verify(prediction: str, gold: str, expected: bool) -> None:
    r"""FiniteSet (非 Equality) の \pm 統合が正しく動作することを確認する（回帰テスト）。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected
