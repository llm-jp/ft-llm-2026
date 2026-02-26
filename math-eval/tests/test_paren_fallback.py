"""test_paren_fallback.py

括弧の種類 (), [], {}, を相互に置き換えるフォールバック評価を確認する。
"""

import pytest

from math_eval.main import parse_and_verify, _paren_variants


# --- _paren_variants 単体テスト ---


class TestParenVariants:
    """_paren_variants が正しくバリエーションを生成することを確認する。"""

    def test_left_right_square(self):
        r"""\left[...\right] → \left(...\right) と \left\{...\right\} を生成。"""
        expr = r"$\left[-3, 5\right]$"
        variants = _paren_variants(expr)
        assert r"$\left(-3, 5\right)$" in variants
        assert r"$\left\{-3, 5\right\}$" in variants
        assert expr not in variants

    def test_left_right_round(self):
        r"""\left(...\right) → \left[...\right] と \left\{...\right\} を生成。"""
        expr = r"$\left(-3, 5\right)$"
        variants = _paren_variants(expr)
        assert r"$\left[-3, 5\right]$" in variants
        assert r"$\left\{-3, 5\right\}$" in variants

    def test_left_right_curly(self):
        r"""\left\{...\right\} → \left(...\right) と \left[...\right] を生成。"""
        expr = r"$\left\{-3, 5\right\}$"
        variants = _paren_variants(expr)
        assert r"$\left(-3, 5\right)$" in variants
        assert r"$\left[-3, 5\right]$" in variants

    def test_plain_round(self):
        """(...) → [...] と \\{...\\} を生成。"""
        expr = r"$(-3, 5)$"
        variants = _paren_variants(expr)
        assert r"$[-3, 5]$" in variants
        assert r"$\{-3, 5\}$" in variants

    def test_plain_square(self):
        """[...] → (...) と \\{...\\} を生成。"""
        expr = r"$[-3, 5]$"
        variants = _paren_variants(expr)
        assert r"$(-3, 5)$" in variants
        assert r"$\{-3, 5\}$" in variants

    def test_no_brackets(self):
        """括弧がない場合は空リストを返す。"""
        assert _paren_variants(r"$3 + 5$") == []
        assert _paren_variants(r"42") == []

    def test_display_math_delimiters_preserved(self):
        r"""数式デリミタ \[, \] は置換対象外。"""
        expr = r"\[-3 + 5\]"
        variants = _paren_variants(expr)
        # \[ \] はデリミタなので置換されない（括弧として扱われない）
        for v in variants:
            assert r"\[" in v or r"\(" not in v

    def test_inline_math_delimiters_preserved(self):
        r"""数式デリミタ \(, \) は置換対象外。"""
        expr = r"\(x + 1\)"
        variants = _paren_variants(expr)
        for v in variants:
            assert r"\(" in v


# --- 括弧フォールバック統合テスト ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # FiniteSet (±) vs Interval ([...]) → 括弧変換で一致
        (
            r"$\left[-3 - \sqrt{6},\ -3 + \sqrt{6}\right]$",
            r"\[-3 \pm \sqrt{6}\]",
            True,
        ),
        # FiniteSet (±) vs open interval ((...)) → 括弧変換で一致
        (
            r"$\left(-3 - \sqrt{6},\ -3 + \sqrt{6}\right)$",
            r"\[-3 \pm \sqrt{6}\]",
            True,
        ),
        # FiniteSet ({...}) vs FiniteSet (±) → そのまま一致
        (
            r"$\left\{-3 - \sqrt{6},\ -3 + \sqrt{6}\right\}$",
            r"\[-3 \pm \sqrt{6}\]",
            True,
        ),
        # plain 括弧: [a, b] vs {a, b}
        (r"$[2, 4]$", r"$\{2, 4\}$", True),
        # plain 括弧: (a, b) vs {a, b}
        (r"$(2, 4)$", r"$\{2, 4\}$", True),
        # 値が異なる場合はフォールバックしても不一致
        (
            r"$\left[-3 - \sqrt{6},\ -3 + \sqrt{5}\right]$",
            r"\[-3 \pm \sqrt{6}\]",
            False,
        ),
        # 単一値（括弧なし）は影響なし
        (r"$5$", r"$5$", True),
        (r"$5$", r"$6$", False),
    ],
)
def test_paren_fallback(prediction: str, gold: str, expected: bool) -> None:
    """括弧の種類が異なる場合のフォールバック評価を確認する。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected
