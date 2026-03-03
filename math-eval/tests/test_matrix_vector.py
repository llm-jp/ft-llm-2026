r"""test_matrix_vector.py

行列ベクトル ↔ タプルの同一視テスト。
縦ベクトル（\begin{pmatrix}...\end{pmatrix}）と横ベクトル（(a, b, c)）を
区別せず同一と判定する。
"""

import pytest

from math_eval.main import _to_flat_vector, parse_and_verify
from sympy import Matrix, Tuple as STuple, Interval, Integer


# --- _to_flat_vector 単体テスト ---


class TestToFlatVector:
    """_to_flat_vector の単体テスト。"""

    def test_column_vector(self) -> None:
        m = Matrix([[1], [2], [3]])
        assert _to_flat_vector(m) == [Integer(1), Integer(2), Integer(3)]

    def test_row_vector(self) -> None:
        m = Matrix([[1, 2, 3]])
        assert _to_flat_vector(m) == [Integer(1), Integer(2), Integer(3)]

    def test_tuple(self) -> None:
        t = STuple(1, 2, 3)
        assert _to_flat_vector(t) == [Integer(1), Integer(2), Integer(3)]

    def test_interval_as_2elem(self) -> None:
        iv = Interval(1, 2, left_open=True, right_open=True)
        assert _to_flat_vector(iv) == [Integer(1), Integer(2)]

    def test_general_matrix_returns_none(self) -> None:
        """2x2 以上の一般行列は None を返す。"""
        m = Matrix([[1, 2], [3, 4]])
        assert _to_flat_vector(m) is None

    def test_scalar_returns_none(self) -> None:
        assert _to_flat_vector(Integer(5)) is None


# --- 統合テスト ---


@pytest.mark.parametrize(
    "prediction, gold, expected",
    [
        # 縦ベクトル vs タプル (3要素)
        (
            r"$(1, 2, 3)$",
            r"$\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$",
            True,
        ),
        # 縦ベクトル vs タプル (逆方向)
        (
            r"$\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$",
            r"$(1, 2, 3)$",
            True,
        ),
        # pmatrix vs bmatrix（verify が直接処理）
        (
            r"$\begin{pmatrix} 1 \\ 2 \end{pmatrix}$",
            r"$\begin{bmatrix} 1 \\ 2 \end{bmatrix}$",
            True,
        ),
        # 行ベクトル vs 縦ベクトル
        (
            r"$\begin{pmatrix} 1 & 2 & 3 \end{pmatrix}$",
            r"$\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$",
            True,
        ),
        # 2要素: 縦ベクトル vs (a, b)（Interval として解釈されるケース）
        (
            r"$(1, 2)$",
            r"$\begin{pmatrix} 1 \\ 2 \end{pmatrix}$",
            True,
        ),
        # 分数を含む縦ベクトル
        (
            r"$\left(\frac{1}{2}, \frac{3}{4}, 1\right)$",
            r"$\begin{pmatrix} \frac{1}{2} \\ \frac{3}{4} \\ 1 \end{pmatrix}$",
            True,
        ),
        # 不一致: 要素が異なる
        (
            r"$(1, 2, 3)$",
            r"$\begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix}$",
            False,
        ),
        # 不一致: 要素数が異なる
        (
            r"$(1, 2)$",
            r"$\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$",
            False,
        ),
    ],
)
def test_matrix_vector_integration(
    prediction: str, gold: str, expected: bool
) -> None:
    """行列ベクトル ↔ タプルの同一視が parse_and_verify で正しく機能することを確認。"""
    result = parse_and_verify(prediction, gold, evaluation_method="soft")
    assert result == expected
