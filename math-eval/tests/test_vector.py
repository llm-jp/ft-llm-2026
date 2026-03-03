r"""test_vector.py

ベクトル記号の正規化テスト。
- _normalize_vector_notation: 全バリアント → \vec{...} への統一
- _strip_vector_notation: 単一文字のみ除去、複数文字は保持
- _verify_vector_fallback: gold のベクトルトークンを pred にも適用
- parse_and_verify 統合テスト
"""

import pytest

from math_eval.main import (
    _normalize_vector_notation,
    _strip_vector_notation,
    _has_vector_notation,
    _extract_vector_names,
    parse_and_verify,
)


# --- _normalize_vector_notation 単体テスト ---


class TestNormalizeVectorNotation:
    r"""_normalize_vector_notation の単体テスト。全バリアント → \vec{...}。"""

    def test_vec_unchanged(self) -> None:
        assert _normalize_vector_notation(r"\vec{a}") == r"\vec{a}"

    def test_vec_without_braces(self) -> None:
        assert _normalize_vector_notation(r"\vec a") == r"\vec{a}"

    def test_overrightarrow(self) -> None:
        assert _normalize_vector_notation(r"\overrightarrow{AB}") == r"\vec{AB}"

    def test_boldsymbol(self) -> None:
        assert _normalize_vector_notation(r"\boldsymbol{a}") == r"\vec{a}"

    def test_mathbf(self) -> None:
        assert _normalize_vector_notation(r"\mathbf{F}") == r"\vec{F}"

    def test_bm(self) -> None:
        assert _normalize_vector_notation(r"\bm{v}") == r"\vec{v}"

    def test_unicode_combining_arrow(self) -> None:
        assert _normalize_vector_notation("a\u20D7") == r"\vec{a}"

    def test_unicode_combining_harpoon(self) -> None:
        assert _normalize_vector_notation("a\u20D1") == r"\vec{a}"

    def test_multiple_vectors(self) -> None:
        assert _normalize_vector_notation(r"\vec{a} + \vec{b}") == r"\vec{a} + \vec{b}"

    def test_no_vector(self) -> None:
        assert _normalize_vector_notation("x + y") == "x + y"


# --- _strip_vector_notation 単体テスト ---


class TestStripVectorNotation:
    """_strip_vector_notation の単体テスト（単一文字のみ除去）。"""

    def test_single_letter_stripped(self) -> None:
        assert _strip_vector_notation(r"\vec{a}") == "a"

    def test_single_letter_without_braces(self) -> None:
        assert _strip_vector_notation(r"\vec a") == "a"

    def test_multi_letter_preserved(self) -> None:
        r"""複数文字の \vec{AB} は保持される（A*B と誤解されないよう）。"""
        assert _strip_vector_notation(r"\overrightarrow{AB}") == r"\vec{AB}"

    def test_boldsymbol_single(self) -> None:
        assert _strip_vector_notation(r"\boldsymbol{a}") == "a"

    def test_mathbf_single(self) -> None:
        assert _strip_vector_notation(r"\mathbf{F}") == "F"

    def test_bm_single(self) -> None:
        assert _strip_vector_notation(r"\bm{v}") == "v"

    def test_unicode_single(self) -> None:
        assert _strip_vector_notation("a\u20D7") == "a"

    def test_multiple_single_letters(self) -> None:
        assert _strip_vector_notation(r"\vec{a} + \vec{b}") == "a + b"

    def test_no_vector(self) -> None:
        assert _strip_vector_notation("x + y") == "x + y"


# --- ユーティリティ関数テスト ---


class TestVectorUtilities:
    """_has_vector_notation, _extract_vector_names のテスト。"""

    def test_has_vec(self) -> None:
        assert _has_vector_notation(r"\vec{a}")

    def test_has_overrightarrow(self) -> None:
        assert _has_vector_notation(r"\overrightarrow{AB}")

    def test_has_unicode(self) -> None:
        assert _has_vector_notation("a\u20D7")

    def test_no_vector(self) -> None:
        assert not _has_vector_notation("x + y")

    def test_extract_names(self) -> None:
        expr = _normalize_vector_notation(r"\vec{a} + \overrightarrow{AB}")
        assert _extract_vector_names(expr) == {"a", "AB"}


# --- 統合テスト ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # --- 単一文字ベクトル（_extended_parse で除去） ---
        # \vec vs 細字
        (r"$\vec{a}$", r"$a$", True),
        (r"$\vec{a} + \vec{b}$", r"$a + b$", True),
        (r"$3\vec{a}$", r"$3a$", True),
        # 太字 vs 細字
        (r"$\boldsymbol{a}$", r"$a$", True),
        (r"$\mathbf{v}$", r"$v$", True),
        (r"$\bm{a} + \bm{b}$", r"$a + b$", True),
        # \vec vs 太字（両方 \vec{a} に統一されて一致）
        (r"$\vec{a}$", r"$\boldsymbol{a}$", True),
        # --- 複数文字ベクトル（フォールバックで処理） ---
        # \overrightarrow{AB} vs bare AB（gold にベクトル記号あり → pred の AB も統一）
        (r"$AB$", r"$\overrightarrow{AB}$", True),
        # 両方にベクトル記号あり
        (r"$\vec{AB}$", r"$\overrightarrow{AB}$", True),
        # 複数文字ベクトルの演算
        (r"$AB + CD$", r"$\overrightarrow{AB} + \overrightarrow{CD}$", True),
        # pred 側にベクトル記号、gold は bare（逆方向）
        (r"$\vec{AB}$", r"$AB$", True),
        # \frac 内の bare ベクトルも1シンボル化
        (
            r"$\frac{AB}{2} + \frac{CD}{3}$",
            r"$\frac{\overrightarrow{AB}}{2} + \frac{\overrightarrow{CD}}{3}$",
            True,
        ),
        # スカラー倍の複数文字ベクトル
        (r"$3AB$", r"$3\overrightarrow{AB}$", True),
        # --- 不一致 ---
        (r"$\vec{a}$", r"$b$", False),
        (r"$\overrightarrow{AB}$", r"$\overrightarrow{CD}$", False),
    ],
)
def test_vector_integration(
    prediction: str, gold: str, expected: bool
) -> None:
    """ベクトル記号の正規化・フォールバックが parse_and_verify で正しく機能することを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected
