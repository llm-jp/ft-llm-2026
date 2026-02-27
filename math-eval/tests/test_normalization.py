r"""test_normalization.py

正規化処理の単体テスト。
- XML タグ除去
- 数値トークン間スペース除去
- LaTeX スペーシングコマンド除去
- 不等号の正規化 (\geqq → \geq 等)
- 論理記号の正規化 (\vee, \lor 等 → カンマ)
- 複数数式ブロックのマージ
"""

import pytest

from math_eval.main import (
    _strip_xml_tags,
    _normalize_digit_spaces,
    _strip_latex_spacing,
    _normalize_inequalities,
    _normalize_logical_connectives,
    _merge_math_blocks,
    parse_and_verify,
)


# --- XML タグ除去 ---


class TestStripXmlTags:
    """_strip_xml_tags の単体テスト。"""

    def test_answer_tag(self):
        assert _strip_xml_tags("<answer>42</answer>") == "42"

    def test_nested_tags(self):
        assert _strip_xml_tags("<div><p>hello</p></div>") == "hello"

    def test_tag_with_attributes(self):
        assert _strip_xml_tags('<answer type="math">x+1</answer>') == "x+1"

    def test_no_tags(self):
        assert _strip_xml_tags("42") == "42"

    def test_self_closing_not_matched(self):
        """自己閉じタグは現在の正規表現では対象外。"""
        assert _strip_xml_tags("$x$") == "$x$"

    def test_latex_not_affected(self):
        r"""LaTeX のコマンドはタグとして除去されない。"""
        assert _strip_xml_tags(r"$\frac{1}{2}$") == r"$\frac{1}{2}$"


# --- 数値トークン間スペース除去 ---


class TestNormalizeDigitSpaces:
    """_normalize_digit_spaces の単体テスト。"""

    def test_integer_spaces(self):
        assert _normalize_digit_spaces("1 2 3") == "123"

    def test_decimal_space(self):
        assert _normalize_digit_spaces("0. 7") == "0.7"

    def test_decimal_space_before_dot(self):
        assert _normalize_digit_spaces("3 .14") == "3.14"

    def test_no_spaces(self):
        assert _normalize_digit_spaces("123") == "123"

    def test_non_digit_spaces_preserved(self):
        assert _normalize_digit_spaces("x + 1") == "x + 1"

    def test_mixed(self):
        assert _normalize_digit_spaces("x = 1 2 3") == "x = 123"


# --- LaTeX スペーシングコマンド除去 ---


class TestStripLatexSpacing:
    r"""_strip_latex_spacing の単体テスト。"""

    def test_thin_space(self):
        assert _strip_latex_spacing(r"a\,b") == "a b"

    def test_medium_space(self):
        assert _strip_latex_spacing(r"a\;b") == "a b"

    def test_thick_space(self):
        assert _strip_latex_spacing(r"a\:b") == "a b"

    def test_negative_space(self):
        assert _strip_latex_spacing(r"a\!b") == "a b"

    def test_quad(self):
        assert _strip_latex_spacing(r"a\quad b") == "a  b"

    def test_qquad(self):
        assert _strip_latex_spacing(r"a\qquad b") == "a  b"

    def test_backslash_space(self):
        r"""\ (バックスラッシュ+スペース) → スペースに。元のスペースと合わせて1スペース。"""
        assert _strip_latex_spacing(r"a\ b") == "a b"

    def test_no_spacing_commands(self):
        assert _strip_latex_spacing(r"a + b") == "a + b"

    def test_latex_commands_not_affected(self):
        r"""LaTeX コマンド (\sin, \sqrt 等) は除去されない。"""
        assert _strip_latex_spacing(r"\sin x") == r"\sin x"
        assert _strip_latex_spacing(r"\sqrt{2}") == r"\sqrt{2}"

    def test_comma_separated_with_spacing(self):
        r"""a > 0,\ b < 0 → a > 0, b < 0 (\ +スペースがスペース1つに)。"""
        result = _strip_latex_spacing(r"a > 0,\ b < 0")
        assert result == "a > 0, b < 0"


# --- 不等号の正規化 ---


class TestNormalizeInequalities:
    r"""_normalize_inequalities の単体テスト。"""

    def test_geqq_to_geq(self):
        assert _normalize_inequalities(r"x \geqq 0") == r"x \geq 0"

    def test_leqq_to_leq(self):
        assert _normalize_inequalities(r"x \leqq 0") == r"x \leq 0"

    def test_neqq_to_neq(self):
        assert _normalize_inequalities(r"x \neqq 0") == r"x \neq 0"

    def test_multiple(self):
        assert _normalize_inequalities(r"a \geqq 0, b \leqq 1") == r"a \geq 0, b \leq 1"

    def test_no_change(self):
        assert _normalize_inequalities(r"x \geq 0") == r"x \geq 0"


# --- 論理記号の正規化 ---


class TestNormalizeLogicalConnectives:
    r"""_normalize_logical_connectives の単体テスト。"""

    def test_vee(self):
        assert _normalize_logical_connectives(r"x < 0 \vee x > 1") == "x < 0 , x > 1"

    def test_lor(self):
        assert _normalize_logical_connectives(r"x < 0 \lor x > 1") == "x < 0 , x > 1"

    def test_wedge(self):
        assert _normalize_logical_connectives(r"x > 0 \wedge x < 1") == "x > 0 , x < 1"

    def test_land(self):
        assert _normalize_logical_connectives(r"x > 0 \land x < 1") == "x > 0 , x < 1"

    def test_text_or(self):
        assert _normalize_logical_connectives(r"a \text{or} b") == "a , b"

    def test_text_and(self):
        assert _normalize_logical_connectives(r"a \text{and} b") == "a , b"

    def test_text_mataha(self):
        assert _normalize_logical_connectives(r"a \text{または} b") == "a , b"

    def test_text_katsu(self):
        assert _normalize_logical_connectives(r"a \text{かつ} b") == "a , b"

    def test_english_or(self):
        assert _normalize_logical_connectives("a or b") == "a , b"

    def test_english_and(self):
        assert _normalize_logical_connectives("a and b") == "a , b"

    def test_double_comma_cleanup(self):
        r"""「a, \text{and} b」パターンで二重カンマが単一カンマになる。"""
        result = _normalize_logical_connectives(r"a, \text{and} b")
        assert ",," not in result
        assert result == "a, b"

    def test_no_change(self):
        assert _normalize_logical_connectives("x < 0, x > 1") == "x < 0, x > 1"


# --- 複数数式ブロックのマージ ---


class TestMergeMathBlocks:
    """_merge_math_blocks の単体テスト。"""

    def test_period_separator(self):
        assert _merge_math_blocks(r"$a$. $b$") == r"$a, b$"

    def test_comma_separator(self):
        assert _merge_math_blocks(r"$a$, $b$") == r"$a, b$"

    def test_or_separator(self):
        assert _merge_math_blocks(r"$a$ or $b$") == r"$a, b$"

    def test_and_separator(self):
        assert _merge_math_blocks(r"$a$ and $b$") == r"$a, b$"

    def test_mataha_separator(self):
        assert _merge_math_blocks(r"$a$ または $b$") == r"$a, b$"

    def test_katsu_separator(self):
        assert _merge_math_blocks(r"$a$ かつ $b$") == r"$a, b$"

    def test_space_only_separator(self):
        assert _merge_math_blocks(r"$a$ $b$") == r"$a, b$"

    def test_three_blocks(self):
        assert _merge_math_blocks(r"$a$. $b$. $c$") == r"$a, b, c$"

    def test_display_math_blocks(self):
        r"""\[...\] ブロックのマージ。"""
        assert _merge_math_blocks(r"\[a\]. \[b\]") == r"\[a, b\]"

    def test_display_math_or(self):
        r"""\[...\] ブロックの or 区切り。"""
        assert _merge_math_blocks(r"\[a\] or \[b\]") == r"\[a, b\]"

    def test_single_block_no_change(self):
        expr = r"$x < -2, 1 < x$"
        assert _merge_math_blocks(expr) == expr

    def test_no_math_no_change(self):
        assert _merge_math_blocks("hello world") == "hello world"


# --- 正規化の統合テスト (parse_and_verify 経由) ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # XML タグ除去
        (r"<answer>$5$</answer>", r"$5$", True),
        (r"<answer>$\sqrt{2}$</answer>", r"$\sqrt{2}$", True),
        # 数値間スペース
        (r"$0. 7$", r"$0.7$", True),
        (r"$1 2 3$", r"$123$", True),
        # LaTeX スペーシング (gold 側)
        (r"$a > 0, b < 0$", r"\[a > 0,\ b < 0\]", True),
        (r"$x$", r"$x\quad$", True),
        # 不等号正規化
        (r"$x \geqq 0$", r"$x \geq 0$", True),
        # 論理記号 + LaTeX スペーシング
        (
            r"\boxed{a > 0,\ b < 0,\ \text{and}\ 4a^2 + b > 0}",
            r"\[a > 0, - 4a^{2} < b < 0\]",
            True,
        ),
        # 複数数式ブロック
        (
            r"$x < -2$. $1 < x$",
            r"$(-\infty, -2) \cup (1, \infty)$",
            True,
        ),
    ],
)
def test_normalization_integration(
    prediction: str, gold: str, expected: bool
) -> None:
    """正規化処理が parse_and_verify を通じて正しく機能することを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected
