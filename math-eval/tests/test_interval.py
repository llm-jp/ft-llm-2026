r"""test_interval.py

区間 (Interval) / 不等式 / 論理記号に関するテスト。
- 不等式 ↔ 区間表記の相互比較
- \vee, \lor, \wedge, \land の正規化
- 複数 $...$ ブロックのマージ
- FiniteSet ↔ Interval の括弧フォールバック許容
"""

import pytest

from math_eval.main import parse_and_verify, _merge_math_blocks


# --- 不等式 ↔ 区間表記 ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # カンマ区切り不等式 ↔ Union(Interval)
        (
            r"$(-\infty, -2) \cup (-1, \infty)$",
            r"\[x < -2, -1 < x\]",
            True,
        ),
        (
            r"$x < -2, -1 < x$",
            r"$(-\infty, -2) \cup (-1, \infty)$",
            True,
        ),
        # 象徴的な値を含む不等式 ↔ 区間（シンプル版）
        (
            r"$x < -\sqrt{2}, \sqrt{2} < x$",
            r"$(-\infty, -\sqrt{2}) \cup (\sqrt{2}, \infty)$",
            True,
        ),
        # 変数を含む不等式 ↔ 区間
        (
            r"$x < -a, a < x$",
            r"$(-\infty, -a) \cup (a, \infty)$",
            True,
        ),
        # 不等式同士（同一）
        (
            r"$x < -2, 1 < x$",
            r"$x < -2, 1 < x$",
            True,
        ),
        # 値が異なる場合は不一致
        (
            r"$(-\infty, -2) \cup (1, \infty)$",
            r"$(-\infty, -3) \cup (2, \infty)$",
            False,
        ),
    ],
)
def test_inequality_interval(prediction: str, gold: str, expected: bool) -> None:
    r"""不等式表記と区間表記の相互比較。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected


# --- \vee / \lor ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        (
            r"$x < -2 \vee -1 < x$",
            r"\[x < -2, -1 < x\]",
            True,
        ),
        (
            r"$x < -2 \lor 1 < x$",
            r"$(-\infty, -2) \cup (1, \infty)$",
            True,
        ),
        (
            r"$x < -2 \vee -1 < x$",
            r"$(-\infty, -2) \cup (-1, \infty)$",
            True,
        ),
    ],
)
def test_vee_lor(prediction: str, gold: str, expected: bool) -> None:
    r"""\vee / \lor の正規化。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected


# --- \wedge / \land ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # \wedge = AND → 区間の共通部分
        (
            r"$x > -2 \wedge x < 1$",
            r"$(-2, 1)$",
            True,
        ),
        (
            r"$-2 < x \land x < 1$",
            r"$(-2, 1)$",
            True,
        ),
        # 多変数連立不等式: 連鎖不等式 vs 式変形された個別不等式
        (
            r"$a > 0 \wedge 4 a^{2} + b > 0 \wedge b < 0$",
            r"\[a > 0, - 4a^{2} < b < 0\]",
            True,
        ),
        # 多変数連立不等式: 値が異なる場合は不一致
        (
            r"$a > 0 \wedge 4 a^{2} + b > 0 \wedge b > 0$",
            r"\[a > 0, - 4a^{2} < b < 0\]",
            False,
        ),
    ],
)
def test_wedge_land(prediction: str, gold: str, expected: bool) -> None:
    r"""\wedge / \land の正規化。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected


# --- FiniteSet ↔ Interval の括弧フォールバック (許容) ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # [a, b] (Interval) → {a, b} (FiniteSet) は許容
        (r"$[2, 4]$", r"$\{2, 4\}$", True),
        (r"$(2, 4)$", r"$\{2, 4\}$", True),
        # {a, b} (FiniteSet) → [a, b] (Interval) は許容
        (r"$\{2, 4\}$", r"$[2, 4]$", True),
        (r"$\{2, 4\}$", r"$(2, 4)$", True),
    ],
)
def test_finiteset_interval_fallback(
    prediction: str, gold: str, expected: bool
) -> None:
    """FiniteSet ↔ Interval の括弧フォールバックが許容されることを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected


# --- 複数 $...$ ブロックのマージ ---


class TestMergeMathBlocks:
    """_merge_math_blocks が隣接 $...$ ブロックを正しく統合することを確認する。"""

    def test_period_separator(self):
        """カンマ区切り: $a$. $b$ → $a, b$"""
        assert _merge_math_blocks(r"$x < -2$, $1 < x$") == r"$x < -2, 1 < x$"

    def test_or_separator(self):
        """英語 or 区切り: $a$ or $b$ → $a, b$"""
        assert _merge_math_blocks(r"$x < -2$ or $1 < x$") == r"$x < -2, 1 < x$"

    def test_mataha_separator(self):
        """日本語「または」区切り: $a$ または $b$ → $a, b$"""
        assert _merge_math_blocks(r"$x < -2$ または $1 < x$") == r"$x < -2, 1 < x$"

    def test_space_only_separator(self):
        """空白のみ: $a$ $b$ → $a, b$"""
        assert _merge_math_blocks(r"$x < -2$ $1 < x$") == r"$x < -2, 1 < x$"

    def test_three_blocks(self):
        """3ブロック: $a$. $b$. $c$ → $a, b, c$"""
        assert _merge_math_blocks(r"$a$. $b$. $c$") == r"$a, b, c$"

    def test_no_merge_single_block(self):
        """単一ブロックは変更なし。"""
        expr = r"$x < -2, 1 < x$"
        assert _merge_math_blocks(expr) == expr


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # ピリオド区切り
        (
            r"$x < -2$. $1 < x$",
            r"$(-\infty, -2) \cup (1, \infty)$",
            True,
        ),
        # 「または」区切り
        (
            r"$x < -2$ または $1 < x$",
            r"$(-\infty, -2) \cup (1, \infty)$",
            True,
        ),
        # or 区切り
        (
            r"$x < -2$ or $1 < x$",
            r"$(-\infty, -2) \cup (1, \infty)$",
            True,
        ),
        # 空白のみ区切り
        (
            r"$x < -2$ $1 < x$",
            r"$(-\infty, -2) \cup (1, \infty)$",
            True,
        ),
        # 値が異なる場合は不一致
        (
            r"$x < -2$. $2 < x$",
            r"$(-\infty, -2) \cup (1, \infty)$",
            False,
        ),
    ],
)
def test_multi_block_merge(prediction: str, gold: str, expected: bool) -> None:
    """複数 $...$ ブロックが区切り文字でマージされることを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected
