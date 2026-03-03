"""test_paren_fallback.py

括弧の種類 (), [], {} を相互に置き換えるフォールバック評価を確認する。
"""

import pytest

from math_eval.main import parse_and_verify, _paren_variants


# --- _paren_variants 単体テスト ---


class TestParenVariants:
    """_paren_variants が正しくバリエーションを生成することを確認する。"""

    def test_left_right_square_to_others(self):
        r"""\left[...\right] → \left(...\right) と \left\{...\right\} を生成。"""
        expr = r"$\left[-3, 5\right]$"
        variants = _paren_variants(expr)
        assert r"$\left(-3, 5\right)$" in variants
        assert r"$\left\{-3, 5\right\}$" in variants
        assert expr not in variants
        assert len(variants) == 2

    def test_left_right_round_to_others(self):
        r"""\left(...\right) → \left[...\right] と \left\{...\right\} を生成。"""
        expr = r"$\left(-3, 5\right)$"
        variants = _paren_variants(expr)
        assert r"$\left[-3, 5\right]$" in variants
        assert r"$\left\{-3, 5\right\}$" in variants
        assert expr not in variants

    def test_left_right_curly_to_others(self):
        r"""\left\{...\right\} → \left(...\right) と \left[...\right] を生成。"""
        expr = r"$\left\{-3, 5\right\}$"
        variants = _paren_variants(expr)
        assert r"$\left(-3, 5\right)$" in variants
        assert r"$\left[-3, 5\right]$" in variants
        assert expr not in variants

    def test_plain_round_to_others(self):
        """plain (...) → [...] と \\{...\\} を生成。"""
        expr = r"$(-3, 5)$"
        variants = _paren_variants(expr)
        assert r"$[-3, 5]$" in variants
        assert r"$\{-3, 5\}$" in variants
        assert expr not in variants

    def test_plain_square_to_others(self):
        """plain [...] → (...) と \\{...\\} を生成。"""
        expr = r"$[-3, 5]$"
        variants = _paren_variants(expr)
        assert r"$(-3, 5)$" in variants
        assert r"$\{-3, 5\}$" in variants
        assert expr not in variants

    def test_plain_curly_to_others(self):
        r"""plain \{...\} → (...) と [...] を生成。"""
        expr = r"$\{-3, 5\}$"
        variants = _paren_variants(expr)
        assert r"$(-3, 5)$" in variants
        assert r"$[-3, 5]$" in variants
        assert expr not in variants

    def test_no_brackets_returns_empty(self):
        """括弧がない場合は空リストを返す。"""
        assert _paren_variants(r"$3 + 5$") == []
        assert _paren_variants(r"42") == []
        assert _paren_variants(r"$\sqrt{2}$") == []
        assert _paren_variants(r"$\frac{1}{2}$") == []

    def test_display_math_delimiters_preserved(self):
        r"""数式デリミタ \[, \] は置換対象外。"""
        expr = r"\[-3 + 5\]"
        variants = _paren_variants(expr)
        for v in variants:
            # \[ と \] がそのまま残っている
            assert r"\[" in v
            assert r"\]" in v

    def test_inline_math_delimiters_preserved(self):
        r"""数式デリミタ \(, \) は置換対象外。"""
        expr = r"\(x + 1\)"
        variants = _paren_variants(expr)
        for v in variants:
            assert r"\(" in v
            assert r"\)" in v

    def test_mixed_brackets_and_display_math(self):
        r"""\[ \] (デリミタ) と \left[...\right] (括弧) が混在する場合。"""
        expr = r"\[\left[-3, 5\right]\]"
        variants = _paren_variants(expr)
        for v in variants:
            # \[ \] デリミタは保持される
            assert v.startswith(r"\[")
            assert v.endswith(r"\]")
        # 括弧部分だけ変換される
        assert r"\[\left(-3, 5\right)\]" in variants
        assert r"\[\left\{-3, 5\right\}\]" in variants

    def test_multiple_bracket_pairs(self):
        """複数の括弧ペアが全て同じ型に変換される。"""
        expr = r"$(1, 2) + (3, 4)$"
        variants = _paren_variants(expr)
        assert r"$[1, 2] + [3, 4]$" in variants
        assert r"$\{1, 2\} + \{3, 4\}$" in variants

    def test_nested_function_parens(self):
        """関数の括弧 f(x) を含む式でもバリエーション生成される。"""
        expr = r"$f(x)$"
        variants = _paren_variants(expr)
        # f(x) → f[x] や f\{x\} が生成される（意味的には不正だが、
        # フォールバックとして試行される。verify が False を返すので無害）
        assert len(variants) == 2


# --- 括弧フォールバック統合テスト (prediction 側) ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # --- prediction の括弧変換 ---
        # Interval ([...]) → FiniteSet (±) に一致
        (
            r"$\left[-3 - \sqrt{6},\ -3 + \sqrt{6}\right]$",
            r"\[-3 \pm \sqrt{6}\]",
            True,
        ),
        # open interval ((...)) → FiniteSet (±) に一致
        (
            r"$\left(-3 - \sqrt{6},\ -3 + \sqrt{6}\right)$",
            r"\[-3 \pm \sqrt{6}\]",
            True,
        ),
        # FiniteSet ({...}) → FiniteSet (±) にそのまま一致
        (
            r"$\left\{-3 - \sqrt{6},\ -3 + \sqrt{6}\right\}$",
            r"\[-3 \pm \sqrt{6}\]",
            True,
        ),
        # plain 括弧: [a, b] vs {a, b}
        (r"$[2, 4]$", r"$\{2, 4\}$", True),
        # plain 括弧: (a, b) vs {a, b}
        (r"$(2, 4)$", r"$\{2, 4\}$", True),
        # plain 括弧: {a, b} vs [a, b]
        (r"$\{2, 4\}$", r"$[2, 4]$", True),
        # --- gold 側の括弧変換 ---
        # gold が Interval で pred が FiniteSet
        (r"$\{2, 4\}$", r"$[2, 4]$", True),
        (r"$\{2, 4\}$", r"$(2, 4)$", True),
        # --- 不一致のケース ---
        # 値が異なる場合はフォールバックしても不一致
        (
            r"$\left[-3 - \sqrt{6},\ -3 + \sqrt{5}\right]$",
            r"\[-3 \pm \sqrt{6}\]",
            False,
        ),
        (r"$[2, 5]$", r"$\{2, 4\}$", False),
        # --- 括弧なしの式は影響なし ---
        (r"$5$", r"$5$", True),
        (r"$5$", r"$6$", False),
        (r"$\sqrt{2}$", r"$\sqrt{2}$", True),
    ],
)
def test_paren_fallback(prediction: str, gold: str, expected: bool) -> None:
    """括弧の種類が異なる場合のフォールバック評価を確認する。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected


# --- strict / complex モードでは括弧フォールバックが無効であることの確認 ---


def test_paren_fallback_not_in_strict() -> None:
    """strict モードでは括弧フォールバックが発動しないことを確認する。

    soft のみに実装されているため、strict では括弧の種類が異なると不一致になる。
    """
    # [2, 4] (Interval) vs {2, 4} (FiniteSet): soft なら一致、strict では不一致
    pred = r"$[2, 4]$"
    gold = r"$\{2, 4\}$"
    result = parse_and_verify(pred, gold, evaluation_method="strict")
    assert result is False
