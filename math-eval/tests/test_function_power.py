r"""test_function_power.py

初等関数の冪乗表記に関するパーステスト。
\sin^2 x ↔ (\sin x)^2, \log{x}^3 ≠ (\log x)^3 等。
"""

import pytest

from math_eval.main import parse_and_verify


# --- 冪乗の同一視（一致するケース） ---


@pytest.mark.parametrize(
    "prediction, gold",
    [
        # sin
        (r"$\sin^2 x$", r"$(\sin x)^2$"),
        (r"$\sin^{2} x$", r"$(\sin x)^2$"),
        (r"$\sin^3(x)$", r"$(\sin(x))^3$"),
        (r"$\sin^2\left(x\right)$", r"$(\sin x)^2$"),
        # cos
        (r"$\cos^2 x$", r"$(\cos x)^2$"),
        (r"$\cos^{2}\theta$", r"$(\cos\theta)^2$"),
        (r"$\cos^{2}\left(\theta\right)$", r"$(\cos\theta)^2$"),
        # tan
        (r"$\tan^2 x$", r"$(\tan x)^2$"),
        # log
        (r"$\log^2 x$", r"$(\log x)^2$"),
        (r"${\log x}^3$", r"$(\log x)^3$"),
        # exp
        (r"$\exp^2(x)$", r"$(\exp(x))^2$"),
        # frac の中
        (r"$\frac{\sin^2 x}{2}$", r"$\frac{(\sin x)^2}{2}$"),
    ],
)
def test_function_power_equivalent(prediction: str, gold: str) -> None:
    r"""f^n(x) と (f(x))^n が同一視されることを確認。"""
    assert parse_and_verify(prediction, gold, evaluation_method="soft") is True


# --- 逆三角関数 ---


@pytest.mark.parametrize(
    "prediction, gold",
    [
        (r"$\sin^{-1} x$", r"$\arcsin x$"),
        (r"$\cos^{-1} x$", r"$\arccos x$"),
        (r"$\tan^{-1} x$", r"$\arctan x$"),
    ],
)
def test_inverse_trig(prediction: str, gold: str) -> None:
    r"""f^{-1}(x) が逆関数として正しくパースされることを確認。"""
    assert parse_and_verify(prediction, gold, evaluation_method="soft") is True


# --- 三角関数の恒等式 ---


@pytest.mark.parametrize(
    "prediction, gold",
    [
        (r"$\sin^2 x + \cos^2 x$", r"$1$"),
        (r"$\sin^2\theta + \cos^2\theta$", r"$1$"),
        (r"$\sin^2(x) + \cos^2(x)$", r"$(\sin(x))^2 + (\cos(x))^2$"),
    ],
)
def test_trig_identity(prediction: str, gold: str) -> None:
    """三角関数の恒等式を利用した同一視。"""
    assert parse_and_verify(prediction, gold, evaluation_method="soft") is True


# --- 指数関数の表記揺れ ---


@pytest.mark.parametrize(
    "prediction, gold",
    [
        (r"$e^{2x}$", r"$\exp(2x)$"),
    ],
)
def test_exp_notation(prediction: str, gold: str) -> None:
    r"""e^{...} と \exp(...) の同一視。"""
    assert parse_and_verify(prediction, gold, evaluation_method="soft") is True


# --- 不一致のケース ---


@pytest.mark.parametrize(
    "prediction, gold",
    [
        # sin^2 x ≠ sin(x^2)
        (r"$\sin^2 x$", r"$\sin(x^2)$"),
        # log^2 x ≠ log(x^2)
        (r"$\log^2 x$", r"$\log(x^2)$"),
        # \log{x}^3 = log(x^3) ≠ (log x)^3
        (r"$\log{x}^3$", r"$(\log x)^3$"),
        # \log{\left(x\right)}^3 = log(x^3) ≠ (log x)^3
        (r"$\log{\left(x\right)}^3$", r"$(\log x)^3$"),
    ],
)
def test_function_power_not_equivalent(prediction: str, gold: str) -> None:
    r"""f^n(x) と f(x^n) が区別されることを確認。"""
    assert parse_and_verify(prediction, gold, evaluation_method="soft") is False
