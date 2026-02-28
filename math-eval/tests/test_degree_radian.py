r"""test_degree_radian.py

度数法 ↔ ラジアンの同一視テスト（soft 限定）。
^\circ / ° の値をラジアンに変換して比較する。
"""

import pytest

from math_eval.main import parse_and_verify


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # pred が度数法、gold がラジアン
        (r"$30^\circ$", r"$\frac{\pi}{6}$", True),
        (r"$45^\circ$", r"$\frac{\pi}{4}$", True),
        (r"$60^\circ$", r"$\frac{\pi}{3}$", True),
        (r"$90^\circ$", r"$\frac{\pi}{2}$", True),
        (r"$180^\circ$", r"$\pi$", True),
        (r"$360^\circ$", r"$2\pi$", True),
        # ^{\circ} 形式
        (r"$30^{\circ}$", r"$\frac{\pi}{6}$", True),
        # Unicode ° 形式
        ("$60°$", r"$\frac{\pi}{3}$", True),
        # gold が度数法、pred がラジアン
        (r"$\frac{\pi}{4}$", r"$45^\circ$", True),
        # 両方度数法（直接一致）
        (r"$90^\circ$", r"$90^\circ$", True),
        # \boxed{} 内の ^\circ
        (r"\boxed{45^\circ}", r"$\frac{\pi}{4}$", True),
        # \boxed{} 内の ^{\circ}
        (r"\boxed{60^{\circ}}", r"$\frac{\pi}{3}$", True),
        # 不一致
        (r"$30^\circ$", r"$\frac{\pi}{4}$", False),
        (r"$45^\circ$", r"$\frac{\pi}{3}$", False),
    ],
)
def test_degree_radian(
    prediction: str, gold: str, expected: bool
) -> None:
    """度数法 ↔ ラジアンの同一視が soft モードで正しく機能することを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # strict では度数法変換しない
        (r"$30^\circ$", r"$\frac{\pi}{6}$", False),
        # strict でも同じ値同士は一致
        (r"$90^\circ$", r"$90^\circ$", True),
    ],
)
def test_degree_radian_strict(
    prediction: str, gold: str, expected: bool
) -> None:
    """strict モードでは度数法変換が行われないことを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method="strict")
    assert result == expected
