"""test_list_gold.py

gold が文字列リストの場合、いずれか一致すれば正解と判定されることを確認する。
"""

import pytest

from math_eval.main import parse_and_verify


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # リスト内の最初の候補が一致
        (r"2", ["2", "二"], True),
        # リスト内の2番目の候補が一致
        (r"\boxed{5}", ["five", "5"], True),
        # リスト内のどちらも不一致
        (r"3", ["2", "二"], False),
        # 要素1つのリスト
        (r"42", ["42"], True),
        # 従来の文字列形式（後方互換性）
        (r"1", "1", True),
    ],
)
def test_verify_with_list_gold(prediction: str, gold: str | list[str], expected: bool) -> None:
    """gold が文字列リストの場合、いずれか一致すれば正解と判定されることを確認する。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # strict でもリスト対応
        (r"2", ["2", "3"], True),
        (r"5", ["3", "4"], False),
    ],
)
def test_verify_with_list_gold_strict(prediction: str, gold: list[str], expected: bool) -> None:
    """strict モードでもリスト形式の gold に対応することを確認する。"""
    result = parse_and_verify(prediction, gold, evaluation_method="strict")
    assert result == expected
