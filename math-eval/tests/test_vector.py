r"""test_vector.py

ベクトル・太字記号の除去テスト。
\vec, \overrightarrow, \boldsymbol, \mathbf, \bm, Unicode 結合矢印を除去し、
記号の有無にかかわらず同一と判定する。
"""

import pytest

from math_eval.main import _strip_vector_notation, parse_and_verify


# --- _strip_vector_notation 単体テスト ---


class TestStripVectorNotation:
    """_strip_vector_notation の単体テスト。"""

    def test_vec_with_braces(self) -> None:
        assert _strip_vector_notation(r"\vec{a}") == "a"

    def test_vec_without_braces(self) -> None:
        assert _strip_vector_notation(r"\vec a") == "a"

    def test_overrightarrow(self) -> None:
        assert _strip_vector_notation(r"\overrightarrow{AB}") == "AB"

    def test_boldsymbol(self) -> None:
        assert _strip_vector_notation(r"\boldsymbol{a}") == "a"

    def test_mathbf(self) -> None:
        assert _strip_vector_notation(r"\mathbf{F}") == "F"

    def test_bm(self) -> None:
        assert _strip_vector_notation(r"\bm{v}") == "v"

    def test_unicode_combining_arrow(self) -> None:
        assert _strip_vector_notation("a\u20D7") == "a"

    def test_unicode_combining_harpoon(self) -> None:
        assert _strip_vector_notation("a\u20D1") == "a"

    def test_multiple_vectors(self) -> None:
        assert _strip_vector_notation(r"\vec{a} + \vec{b}") == "a + b"

    def test_no_vector(self) -> None:
        assert _strip_vector_notation("x + y") == "x + y"


# --- 統合テスト ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # \vec vs 細字
        (r"$\vec{a}$", r"$a$", True),
        (r"$\vec{a} + \vec{b}$", r"$a + b$", True),
        (r"$3\vec{a}$", r"$3a$", True),
        # \overrightarrow vs 細字
        (r"$\overrightarrow{AB}$", r"$AB$", True),
        # 太字 vs 細字
        (r"$\boldsymbol{a}$", r"$a$", True),
        (r"$\mathbf{v}$", r"$v$", True),
        (r"$\bm{a} + \bm{b}$", r"$a + b$", True),
        # \vec vs 太字（両方除去されて一致）
        (r"$\vec{a}$", r"$\boldsymbol{a}$", True),
        # 不一致
        (r"$\vec{a}$", r"$b$", False),
    ],
)
def test_vector_integration(
    prediction: str, gold: str, expected: bool
) -> None:
    """ベクトル記号除去が parse_and_verify で正しく機能することを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected
