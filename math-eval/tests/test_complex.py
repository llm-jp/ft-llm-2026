r"""test_complex.py

complex モード（シンボル i → 虚数単位 I 変換）のテスト。
- 基本的な複素数の等価性
- \pm/\mp と複素数の組み合わせ
"""

import pytest

from math_eval.main import parse_and_verify


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        (r"$1/i$", r"$-i$", True),
        (r"$e^{i\theta} + e^{-i\theta}$", r"$2\cos\theta$", True),
        (r"$e^{ix} / (\cos x + i \sin x)$", "1", True),
        (r"$e^{ix} * (\cos x - i \sin x)$", "1", True),
        (r"$e^{ix}$", r"$1/e^{-ix}$", True),
        (r"$e^{ix}$", r"$\cos x + i \sin x$", True),
        (r"$e^{i\pi/2}$", r"$i$", True),
    ],
)
def test_verify(prediction: str, gold: str, expected: bool) -> None:
    result = parse_and_verify(prediction, gold, evaluation_method="complex")
    assert result == expected


# --- 複素数 + \pm/\mp のテスト ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # \pm で展開される複素数
        (r"$1 \pm 2i$", r"$1 + 2i, 1 - 2i$", True),
        (r"$1 \pm 2i$", r"$1 \mp 2i$", True),
        # 複素数の \pm と明示的な値
        (r"$3 \pm 4i$", r"$3 + 4i, 3 - 4i$", True),
        # 純虚数の \pm
        (r"$\pm i$", r"$i, -i$", True),
        (r"$\pm 3i$", r"$3i, -3i$", True),
        # 値が異なる場合は不一致
        (r"$1 \pm 2i$", r"$1 + 3i, 1 - 3i$", False),
        (r"$1 \pm 2i$", r"$2 + 2i, 2 - 2i$", False),
    ],
)
def test_complex_pm(prediction: str, gold: str, expected: bool) -> None:
    r"""複素数 + \pm/\mp の complex モードでの検証。"""
    result = parse_and_verify(prediction, gold, evaluation_method="complex")
    assert result == expected


# --- バグ修正の回帰テスト ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # Bug1: i を含まない式での二重デリミタ
        (r"$\frac{\pi}{4}$", r"\[\frac{\pi}{4}\]", True),
        (r"$-1$", r"$-1$", True),
        # Bug2: FiniteSet に対する .expand() クラッシュ
        (r"$x = -3, 1 - 2i, 1 + 2i$", r"\[x = - 3,1 \pm 2i\]", True),
        (r"$\{-3, 1 \pm 2i\}$", r"\[x = - 3,1 \pm 2i\]", True),
        # Bug3: simplify() による形の不一致 (極形式 vs 指数形式)
        (
            r"$2 e^{\frac{5 i \pi}{6}}$",
            r"$2\left( \cos\frac{5}{6}\pi + i\sin\frac{5}{6}\pi \right)$",
            True,
        ),
        # 基本的な複素数 (回帰テスト)
        (r"$\frac{11+3i}{5}$", r"$\frac{11+3i}{5}$", True),
        (r"$4 - 4i$", r"$4 - 4i$", True),
        (r"$-i$", r"$-i$", True),
    ],
)
def test_complex_bugfix(prediction: str, gold: str, expected: bool) -> None:
    """バグ報告に基づく回帰テスト。"""
    result = parse_and_verify(prediction, gold, evaluation_method="complex")
    assert result == expected
