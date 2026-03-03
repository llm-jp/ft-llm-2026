r"""test_bare_keywords.py

エスケープされていない数学キーワードの LaTeX コマンドへの変換テスト。
pi → \pi, sin → \sin, cos → \cos 等。
"""

import pytest

from math_eval.main import _bare_keywords_to_latex, parse_and_verify


# --- _bare_keywords_to_latex 単体テスト ---


class TestBareKeywordsToLatex:
    r"""_bare_keywords_to_latex の単体テスト。"""

    def test_pi(self) -> None:
        assert _bare_keywords_to_latex("pi") == r"\pi"

    def test_2pi(self) -> None:
        assert _bare_keywords_to_latex("2pi") == r"2\pi"

    def test_pi_over_6(self) -> None:
        assert _bare_keywords_to_latex("pi/6") == r"\pi/6"

    def test_sin(self) -> None:
        assert _bare_keywords_to_latex("sin(x)") == r"\sin(x)"

    def test_cos(self) -> None:
        assert _bare_keywords_to_latex("cos(x)") == r"\cos(x)"

    def test_tan(self) -> None:
        assert _bare_keywords_to_latex("tan(x)") == r"\tan(x)"

    def test_log(self) -> None:
        assert _bare_keywords_to_latex("log(x)") == r"\log(x)"

    def test_ln(self) -> None:
        assert _bare_keywords_to_latex("ln(2)") == r"\ln(2)"

    def test_exp(self) -> None:
        assert _bare_keywords_to_latex("exp(1)") == r"\exp(1)"

    def test_sqrt(self) -> None:
        assert _bare_keywords_to_latex("sqrt(2)") == r"\sqrt(2)"

    def test_combined(self) -> None:
        """複数キーワードの同時変換。"""
        assert _bare_keywords_to_latex("cos(2pi)") == r"\cos(2\pi)"

    def test_already_escaped_pi(self) -> None:
        r"""既に \pi の場合は変換しない。"""
        assert _bare_keywords_to_latex(r"\pi") == r"\pi"

    def test_already_escaped_sin(self) -> None:
        r"""既に \sin の場合は変換しない。"""
        assert _bare_keywords_to_latex(r"\sin(x)") == r"\sin(x)"

    def test_not_substring_spine(self) -> None:
        """spine の中の pi を変換しない。"""
        assert _bare_keywords_to_latex("spine") == "spine"

    def test_not_substring_pin(self) -> None:
        """pin の中の pi を変換しない。"""
        assert _bare_keywords_to_latex("pin") == "pin"

    def test_no_keywords(self) -> None:
        """キーワードなしの場合は変更なし。"""
        assert _bare_keywords_to_latex("x + y") == "x + y"


# --- 統合テスト ---


@pytest.mark.parametrize(
    "prediction, gold, method, expected",
    [
        # pi → \pi (soft)
        (r"$2pi$", r"$2\pi$", "soft", True),
        (r"$pi/6$", r"$\frac{\pi}{6}$", "soft", True),
        # pi → \pi (strict)
        (r"$2pi$", r"$2\pi$", "strict", True),
        # sin, cos の変換
        (r"$sin(pi/6)$", r"$\sin(\frac{\pi}{6})$", "soft", True),
        (r"$cos(pi/4)$", r"$\frac{\sqrt{2}}{2}$", "soft", True),
        # 不一致
        (r"$2pi$", r"$3\pi$", "soft", False),
        (r"$sin(pi/6)$", r"$\frac{1}{3}$", "soft", False),
    ],
)
def test_bare_keywords_integration(
    prediction: str, gold: str, method: str, expected: bool
) -> None:
    """裸の数学キーワードの変換が parse_and_verify で正しく機能することを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method=method)
    assert result == expected
