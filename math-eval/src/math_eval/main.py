"""Evaluate predictions on mathematical reasoning tasks.

Usage:
    uv run math-eval <prediction-file> <gold-file> [-o <output-file>]
"""

import json
import dataclasses
import re
import warnings
from itertools import product
from typing import Any, Literal
from typing import Optional

import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.table import Table

from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
from latex2sympy2_extended import latex2sympy
from latex2sympy2_extended.math_normalization import NormalizationConfig
from latex2sympy2_extended.sets import FiniteSet as L2SFiniteSet
from sympy import Eq, FiniteSet, I, Interval, Matrix, S, Tuple as STuple, Union, latex, srepr
from sympy.core.relational import Relational

app = typer.Typer()

# Error and Warnings
console = Console()
err_console = Console(stderr=True)


def _custom_warning_format(message, category, filename, lineno, _file=None, _line=None):
    err_console.log(
        f"[yellow]{filename}:{lineno}: {category.__name__}: {message}[/yellow]"
    )


warnings.showwarning = _custom_warning_format

EvaluationMethod = Literal["soft", "strict", "complex", "complex-strict"]


@dataclasses.dataclass
class PredictionExample:
    id: int
    output: str
    problem: Optional[str] = None
    solution: Optional[str] = None
    category: Optional[str] = None
    unit: Optional[str] = None
    difficulty: Optional[str] = None
    evaluation_method: Optional[EvaluationMethod] = None


@dataclasses.dataclass
class GoldExample:
    id: int
    problem: str
    solution: str | list[str]
    category: str
    unit: str
    difficulty: Optional[str] = None
    evaluation_method: Optional[EvaluationMethod] = None


def load_examples(file_path: str, example_cls: type) -> dict[str, Any]:
    """Load examples from a JSONL file."""
    id_example_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                err_console.log(f"Error decoding JSON line in '{file_path}': {e}")
                continue
            try:
                example = example_cls(**item)
            except TypeError as e:
                err_console.log(
                    f"Error creating {example_cls.__name__} from line in '{file_path}': {e}"
                )
                continue
            if example.id in id_example_map:
                err_console.log(
                    f"Duplicate example ID '{example.id}' found in '{file_path}'; overwriting previous entry."
                )
            id_example_map[example.id] = example
    return id_example_map


# 単位除去を無効にした extraction config
# math_verify デフォルトの units=True は ab, bc, pm 等を単位として除去してしまい、
# 数学の変数と衝突するため無効化する
_LATEX_CONFIG_NO_UNITS = LatexExtractionConfig(
    normalization_config=NormalizationConfig(
        basic_latex=True,
        units=False,
        malformed_operators=True,
        nits=True,
        boxed="all",
        equations=False,
    )
)
_EXTRACTION_CONFIG = [_LATEX_CONFIG_NO_UNITS, ExprExtractionConfig()]


# XML タグ除去: LLM 出力に含まれる <answer> 等のタグを除去する
_XML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9_-]*(?:\s[^>]*)?>")


def _strip_xml_tags(text: str) -> str:
    """XML/HTML タグを除去する。"""
    return _XML_TAG_RE.sub("", text)


# 数値トークン間スペース除去: LaTeX 数式モードでは無視されるスペースを正規化
# 例: "0. 7" → "0.7", "1 2 3" → "123"
# ただし添字直後 (_2 3) は除外: \log_2 3 → \log_23 になるのを防ぐ
_DIGIT_SPACE_RE = re.compile(r"(?<=[0-9])(?<!_[0-9])\s+(?=[0-9.])|(?<=[.])\s+(?=[0-9])")


def _normalize_digit_spaces(text: str) -> str:
    """数値トークン間の不要なスペースを除去する。"""
    return _DIGIT_SPACE_RE.sub("", text)


_MATH_DELIMITER_RE = re.compile(
    r"(?<!\\)\$\$"      # $$
    r"|(?<!\\)\\\["     # \[
    r"|(?<!\\)\\\]"     # \]
    r"|(?<!\\|\d)\$"    # $（\$ や 数字$ を除く）
    r"|(?<!\\)\\\("     # \(
    r"|(?<!\\)\\\)"     # \)
    r"|\\boxed\{"       # \boxed{
)


def _ensure_math_delimiters(expr: str) -> str:
    r"""数式デリミタがない場合、$...$ で囲む。

    math_verify の parse は $...$, \(...\), \[...\], $$...$$ などのデリミタを
    正規表現で検出する。デリミタがないとプレーン数値抽出にフォールバックし、
    LaTeX 式（\sqrt{2} など）が正しくパースされない。
    """
    if _MATH_DELIMITER_RE.search(expr):
        return expr
    return f"${expr}$"


# 確率変数の正規化: P(X=k) → p_{Xk} に変換
# P(X=0)=\frac{1}{28} の括弧内の = がパーサーを混乱させるため、
# 変数名に変換してから Eq としてパースさせる
_PROB_VAR_RE = re.compile(r"P\s*\(\s*([A-Za-z])\s*=\s*([^)]+?)\s*\)")


def _normalize_prob_vars(expr: str) -> str:
    r"""確率変数 P(X=k) を p_{Xk} に正規化する。"""
    return _PROB_VAR_RE.sub(
        lambda m: f"p_{{{m.group(1)}{m.group(2).strip()}}}", expr
    )


# ---------------------------------------------------------------------------
# ベクトル記号の正規化
# \overrightarrow, \boldsymbol, \mathbf, \bm, Unicode → \vec{...} に統一
# ---------------------------------------------------------------------------
_VECTOR_NOTATION_RE = re.compile(
    r"\\(?:vec|overrightarrow|boldsymbol|mathbf|bm)\s*"
    r"(?:\{([^}]*)\}|([A-Za-z]))"
)
# Unicode の結合文字によるベクトル記号: a⃗ (U+20D7), a⃑ (U+20D1)
_VECTOR_UNICODE_RE = re.compile(r"([A-Za-z])[\u20D7\u20D1]")
# 残留 Unicode 結合矢印のクリーンアップ用
_VECTOR_UNICODE_CLEANUP_RE = re.compile(r"[\u20D7\u20D1]")
# 正規化済み \vec{...} からトークン名を抽出
_VEC_CONTENT_RE = re.compile(r"\\vec\s*\{([^}]+)\}")


def _normalize_vector_notation(expr: str) -> str:
    r"""全ベクトルバリアントを \vec{...} に統一する。

    \overrightarrow{AB} → \vec{AB}, \boldsymbol{a} → \vec{a},
    \mathbf{F} → \vec{F}, \bm{v} → \vec{v}, a⃗ → \vec{a}
    """
    # LaTeX コマンド → \vec{...}
    expr = _VECTOR_NOTATION_RE.sub(
        lambda m: f"\\vec{{{m.group(1) or m.group(2)}}}", expr
    )
    # Unicode 結合矢印 → \vec{x}
    expr = _VECTOR_UNICODE_RE.sub(r"\\vec{\1}", expr)
    # 残留 Unicode 結合矢印のクリーンアップ
    expr = _VECTOR_UNICODE_CLEANUP_RE.sub("", expr)
    return expr


def _strip_vector_notation(expr: str) -> str:
    r"""ベクトル記号を正規化後、単一文字のみ除去する。

    単一文字: \vec{a} → a（パーサーに安全）
    複数文字: \vec{AB} → そのまま残す（AB が A*B と誤解されるのを防ぐ）
    """
    expr = _normalize_vector_notation(expr)
    # 単一文字ベクトルのみ除去
    expr = re.sub(r"\\vec\s*\{([A-Za-z])\}", r"\1", expr)
    return expr


def _has_vector_notation(expr: str) -> bool:
    """式にベクトル記号が含まれるかチェック。"""
    return bool(_VECTOR_NOTATION_RE.search(expr)) or bool(
        re.search(r"[\u20D7\u20D1]", expr)
    )


def _extract_vector_names(expr: str) -> set[str]:
    r"""正規化済み式から \vec{...} のトークン名を抽出する。"""
    return set(_VEC_CONTENT_RE.findall(expr))


def _verify_vector_fallback(prediction: str, gold: str) -> bool:
    r"""ベクトル記号のフォールバック。

    gold/pred にベクトル記号がある場合、記法を統一して再比較する。
    - 全バリアント → \vec{...} に正規化
    - gold のベクトルトークン名を pred の bare トークンにも適用（逆も同様）
    - 単一文字: \vec{a} → a（除去）
    - 複数文字: \vec{AB} → そのまま保持してパースに渡す
    """
    if not _has_vector_notation(prediction) and not _has_vector_notation(gold):
        return False

    # 全バリアントを \vec{...} に正規化
    norm_gold = _normalize_vector_notation(gold)
    norm_pred = _normalize_vector_notation(prediction)

    # ベクトルトークン名を抽出
    gold_vec_names = _extract_vector_names(norm_gold)
    pred_vec_names = _extract_vector_names(norm_pred)
    all_vec_names = gold_vec_names | pred_vec_names

    if not all_vec_names:
        return False

    # gold のベクトル名を pred の bare トークンにも適用（逆も同様）
    # \vec{AB} の中の AB はマッチしない（負の後読みで保護）
    # \frac{3 AB}{2} などの AB はマッチする
    # NOTE: re.sub の replacement では \v が VT に解釈されるため lambda を使う
    for name in sorted(all_vec_names, key=len, reverse=True):
        bare_pattern = (
            r"(?<!\\vec\{)"  # \vec{...} 内を除外
            + r"(?<![A-Za-z\\])"  # 前が英字や \ でない
            + re.escape(name)
            + r"(?![A-Za-z])"  # 後が英字でない
        )
        vec_wrapped = f"\\vec{{{name}}}"
        if name in gold_vec_names:
            norm_pred = re.sub(bare_pattern, lambda _, w=vec_wrapped: w, norm_pred)
        if name in pred_vec_names:
            norm_gold = re.sub(bare_pattern, lambda _, w=vec_wrapped: w, norm_gold)

    # 単一文字ベクトルを除去（両側を統一してから除去）
    norm_gold = re.sub(r"\\vec\s*\{([A-Za-z])\}", r"\1", norm_gold)
    norm_pred = re.sub(r"\\vec\s*\{([A-Za-z])\}", r"\1", norm_pred)

    # 再パース・再比較
    # \vec{AB} はパーサーが Symbol('vec{ab}') として1シンボルに扱う
    try:
        parsed_gold = _extended_parse(norm_gold)
        parsed_pred = _extended_parse(norm_pred)
        return verify(parsed_gold, parsed_pred)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 行列ベクトル ↔ タプル同一視: 縦ベクトル・横ベクトル・タプルを区別しない
# ---------------------------------------------------------------------------


def _to_flat_vector(val: object) -> list | None:
    """Matrix(列/行ベクトル), Tuple, Interval(2要素) → 要素のフラットリスト。

    一般行列（2行2列以上）は None を返す。
    """
    if isinstance(val, Matrix):
        if val.cols == 1 or val.rows == 1:
            return list(val)
        return None
    if isinstance(val, STuple):
        return list(val)
    if isinstance(val, Interval):
        # (a, b) が開区間として解釈されたケース → 2要素ベクトルとみなす
        return [val.start, val.end]
    return None


def _verify_matrix_vector(parsed_pred: list, parsed_gold: list) -> bool:
    """行列ベクトル ↔ タプルの同一視フォールバック。

    少なくとも片方が Matrix の場合に、要素を展開して比較する。
    """
    p_val = parsed_pred[0] if parsed_pred else None
    g_val = parsed_gold[0] if parsed_gold else None
    if p_val is None or g_val is None:
        return False

    # 少なくとも片方が Matrix でなければ発動しない
    if not isinstance(p_val, Matrix) and not isinstance(g_val, Matrix):
        return False

    p_flat = _to_flat_vector(p_val)
    g_flat = _to_flat_vector(g_val)

    if p_flat is None or g_flat is None:
        return False
    if len(p_flat) != len(g_flat):
        return False

    return all(verify([a], [b]) for a, b in zip(p_flat, g_flat))


# ---------------------------------------------------------------------------
# 小数近似比較: 小数 vs 分数/無理数 を小数の桁数に合わせて丸めて比較
# ---------------------------------------------------------------------------

# 小数の桁数を数えるための正規表現
_DECIMAL_NUM_RE = re.compile(r"-?\d+\.(\d+)")


def _count_decimal_places(s: str) -> int | None:
    """文字列中の小数の桁数を返す（複数ある場合は最小値）。"""
    matches = _DECIMAL_NUM_RE.findall(s)
    if not matches:
        return None
    return min(len(m) for m in matches)


def _verify_numeric_approx(parsed_pred: list, parsed_gold: list) -> bool:
    """小数の近似比較フォールバック。

    片方が小数のとき、小数の桁数に合わせて丸めて比較する。
    例: pred "0.333" vs gold "\\frac{1}{3}" → 小数3桁で丸め → 0.333 == 0.333 → True
    """
    try:
        p_val = parsed_pred[0] if parsed_pred else None
        g_val = parsed_gold[0] if parsed_gold else None
        if p_val is None or g_val is None:
            return False

        # raw string から小数桁数を取得
        p_raw = parsed_pred[1] if len(parsed_pred) > 1 else ""
        g_raw = parsed_gold[1] if len(parsed_gold) > 1 else ""

        p_dec = _count_decimal_places(str(p_raw))
        g_dec = _count_decimal_places(str(g_raw))

        # 少なくとも片方が小数でなければ対象外
        if p_dec is None and g_dec is None:
            return False

        # 両方小数なら桁数が少ない方に合わせる
        if p_dec is not None and g_dec is not None:
            dec_places: int = min(p_dec, g_dec)
        elif p_dec is not None:
            dec_places = p_dec
        else:
            assert g_dec is not None  # 上の None チェックで保証済み
            dec_places = g_dec

        return _approx_equal_values(p_val, g_val, dec_places)
    except Exception:
        return False


def _approx_equal_values(a, b, dec_places: int) -> bool:
    """sympy オブジェクトを数値評価して小数 dec_places 桁で丸めて比較する。"""
    from latex2sympy2_extended.sets import FiniteSet as L2SFiniteSet
    from sympy import FiniteSet as SympyFiniteSet

    # FiniteSet 同士: 要素数が同じなら値をソートして要素ごとに比較
    if isinstance(a, (L2SFiniteSet, SympyFiniteSet)) and isinstance(
        b, (L2SFiniteSet, SympyFiniteSet)
    ):
        try:
            a_vals = sorted(float(x.evalf()) for x in a.args)
            b_vals = sorted(float(x.evalf()) for x in b.args)
        except (TypeError, ValueError):
            return False
        if len(a_vals) != len(b_vals):
            return False
        return all(
            round(x, dec_places) == round(y, dec_places)
            for x, y in zip(a_vals, b_vals)
        )

    # スカラー
    try:
        a_float = float(a.evalf())
        b_float = float(b.evalf())
        return round(a_float, dec_places) == round(b_float, dec_places)
    except (TypeError, ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Dict 形式の正規化: {k1: v1, k2: v2, ...} → 方程式 / 値リスト
# gold・prediction どちらにも適用可能
# ---------------------------------------------------------------------------


def _parse_dict_notation(expr: str) -> list[str] | None:
    r"""Dict 風表記を正規化された数式文字列に変換する。

    {x: 1, y: 2}      → ["$x = 1, y = 2$"]
    {0: 1/28, 1: 3/7}  → ["$\frac{1}{28}, \frac{3}{7}$"]  (数値キーは値のみ)

    対応しないフォーマットの場合は None を返す。
    """
    s = expr.strip()
    # 数式デリミタを除去
    for start, end in [("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)")]:
        if s.startswith(start) and s.endswith(end):
            s = s[len(start) : -len(end)].strip()
            break
    else:
        # $ は 1 文字なので別処理
        if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
            s = s[1:-1].strip()
    # \boxed{...} を除去
    if s.startswith(r"\boxed{") and s.endswith("}"):
        s = s[7:-1].strip()

    # 外側の {} があれば除去（なくても dict 風パターンとして続行）
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    elif s.startswith(r"\{") and s.endswith(r"\}"):
        s = s[2:-2].strip()

    # カンマで分割（{} のネストを考慮）
    parts: list[str] = []
    depth = 0
    current = ""
    for ch in s:
        if ch == "{":
            depth += 1
            current += ch
        elif ch == "}":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())

    # 各パートを key: value に分割（最初の : で分割）
    pairs: list[tuple[str, str]] = []
    for part in parts:
        colon_idx = part.find(":")
        if colon_idx == -1:
            return None
        key = part[:colon_idx].strip()
        value = part[colon_idx + 1 :].strip()
        if not key or not value:
            return None
        pairs.append((key, value))
    if len(pairs) < 2:
        # 単一の key:value は比（2:3 など）と区別できないため dict とみなさない
        return None

    candidates: list[str] = []

    # 全キーが数値の場合は値のみの候補を追加
    all_numeric = all(k.lstrip("-").replace(".", "").isdigit() for k, _ in pairs)
    if all_numeric:
        val_str = ", ".join(v for _, v in pairs)
        candidates.append(f"${val_str}$")
    else:
        # 変数キーの場合は方程式形式
        eq_str = ", ".join(f"{k} = {v}" for k, v in pairs)
        candidates.append(f"${eq_str}$")

    return candidates



# 不等号の正規化: 日本式の二重線 (\geqq 等) を標準形 (\geq 等) に統一
_INEQUALITY_NORMALIZE_MAP = {
    r"\geqq": r"\geq",
    r"\leqq": r"\leq",
    r"\neqq": r"\neq",
    r"\gneqq": r"\gneq",
    r"\lneqq": r"\lneq",
}
_INEQUALITY_RE = re.compile(
    r"|".join(re.escape(k) for k in _INEQUALITY_NORMALIZE_MAP)
)


def _normalize_inequalities(expr: str) -> str:
    r"""日本式の二重線不等号 (\geqq, \leqq 等) を標準形に正規化する。"""
    return _INEQUALITY_RE.sub(
        lambda m: _INEQUALITY_NORMALIZE_MAP[m.group()], expr
    )


# 論理記号の正規化: \vee, \lor, \wedge, \land, \text{または}, or 等を , に置換
# latex2sympy2_extended がこれらをパースできないため、カンマ区切りに変換する
_LOGICAL_CONNECTIVE_RE = re.compile(
    r"\\(?:vee|lor|wedge|land)(?![a-zA-Z])"
    r"|\\(?:text|mathrm|textbf|textit)\s*\{\s*(?:または|かつ|or|and)\s*\}"
    r"|\b(?:or|and)\b",
    re.IGNORECASE,
)


def _normalize_logical_connectives(expr: str) -> str:
    r"""論理記号 (\vee, \lor, \text{または}, or 等) をカンマに正規化する。

    「a, b, \text{and} c」パターンで二重カンマが生じる場合もクリーンアップする。
    """
    expr = _LOGICAL_CONNECTIVE_RE.sub(",", expr)
    # 二重カンマを単一カンマに (「a, and b」→「a,,b」のケース)
    expr = re.sub(r",\s*,", ",", expr)
    return expr


# LaTeX スペーシングコマンドの除去
# \ , \,, \;, \:, \!, \quad, \qquad は視覚的スペースのみで数式に影響しない
_LATEX_SPACING_RE = re.compile(
    r"\\(?:quad|qquad|[,;:!])"
    r"|(?<!\\)\\ "  # \ (バックスラッシュ+スペース)。\sin 等は \+文字 なので非該当
)


def _strip_latex_spacing(expr: str) -> str:
    r"""LaTeX スペーシングコマンド (\,, \;, \quad 等) を通常スペースに変換する。"""
    return _LATEX_SPACING_RE.sub(" ", expr)


# 括弧サイズコマンドの除去
# \bigl, \bigr, \Bigl, \Bigr, \biggl, \biggr, \Biggl, \Biggr,
# \left, \right, \middle は視覚的サイズ指定のみで数式に影響しない
_BRACKET_SIZE_RE = re.compile(
    r"\\(?:big{1,2}|Big{1,2})[lr]"
    r"|\\(?:left|right|middle)"
)


# \log (底なし) ↔ \ln フォールバック用正規表現
# \log_2, \log_{10} 等は底が明示されているので対象外
_LOG_NO_BASE_RE = re.compile(r"\\log(?![_a-zA-Z])")

# ---------------------------------------------------------------------------
# 裸の数学キーワード → LaTeX コマンド変換
# pi → \pi, sin → \sin, cos → \cos, tan → \tan, log → \log, ln → \ln, exp → \exp
# ---------------------------------------------------------------------------
_BARE_MATH_KEYWORDS: dict[str, str] = {
    "pi": r"\pi",
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "log": r"\log",
    "ln": r"\ln",
    "exp": r"\exp",
    "sqrt": r"\sqrt",
}
# 長いキーワードから先にマッチさせる（sqrt が sq + rt に分割されないよう）
_BARE_MATH_RE = re.compile(
    r"(?<![a-zA-Z\\])"
    r"(" + "|".join(sorted(_BARE_MATH_KEYWORDS.keys(), key=len, reverse=True)) + r")"
    r"(?![a-zA-Z])"
)


def _bare_keywords_to_latex(expr: str) -> str:
    r"""裸の数学キーワードを LaTeX コマンドに変換する。

    pi → \pi, sin → \sin, cos → \cos 等。
    2pi → 2\pi, sin(x) → \sin(x) のように文脈を保持する。
    """
    return _BARE_MATH_RE.sub(lambda m: _BARE_MATH_KEYWORDS[m.group(1)], expr)


# ---------------------------------------------------------------------------
# 小数 → 分数 変換: 0.625 → \frac{625}{1000} (パーサが Rational として扱う)
# ---------------------------------------------------------------------------
_DECIMAL_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)(?!\d)")


def _decimal_to_fraction(expr: str) -> str:
    r"""小数を分数に変換する。0.625 → \frac{625}{1000} 等。

    LaTeX パーサが Float ではなく Rational として解釈するようにする。
    \frac{0.625x}{2} や \sin(0.5) 内の小数にも対応。
    """

    def _replace(m: re.Match) -> str:
        int_part = m.group(1)
        dec_part = m.group(2)
        numerator = int(int_part + dec_part)
        denominator = 10 ** len(dec_part)
        return rf"\frac{{{numerator}}}{{{denominator}}}"

    return _DECIMAL_RE.sub(_replace, expr)


# 度数法の検出: ^\circ, ^{\circ}, ° (Unicode)
_DEGREE_RE = re.compile(r"\^\{\\circ\}|\^\\circ|°")


def _verify_degree_radian(prediction: str, gold: str) -> bool:
    r"""度数法 → ラジアン変換フォールバック（soft 限定）。

    ^\circ や ° を \cdot\frac{\pi}{180} に置換してラジアンに変換し、再比較する。
    ラジアン → 度数法への変換は行わない。
    """
    p_has_deg = bool(_DEGREE_RE.search(prediction))
    g_has_deg = bool(_DEGREE_RE.search(gold))

    if not p_has_deg and not g_has_deg:
        return False

    def _deg_to_rad(expr: str) -> str:
        return _DEGREE_RE.sub(r"\\cdot\\frac{\\pi}{180}", expr)

    try:
        if p_has_deg:
            parsed_rad = _extended_parse(_deg_to_rad(prediction))
            parsed_gold = _extended_parse(gold)
            if verify(parsed_rad, parsed_gold):
                return True
        if g_has_deg:
            parsed_pred = _extended_parse(prediction)
            parsed_rad = _extended_parse(_deg_to_rad(gold))
            if verify(parsed_pred, parsed_rad):
                return True
    except Exception:
        pass

    return False


def _strip_bracket_sizing(expr: str) -> str:
    r"""括弧サイズコマンド (\bigl, \bigr, \left, \right 等) を除去する。"""
    return _BRACKET_SIZE_RE.sub("", expr)


def _expand_pm_mp(expr: str) -> list[str]:
    r"""Expand \pm and \mp in expression to generate all combinations.

    \pm expands to + and -
    \mp expands to - and +

    Returns a list of all expanded expressions.
    """
    pattern = re.compile(r"\\(pm|mp)(?![a-zA-Z])")

    matches = list(pattern.finditer(expr))
    if not matches:
        return [expr]

    num_matches = len(matches)
    results = []

    for signs in product(["+", "-"], repeat=num_matches):
        new_expr = expr
        for match, sign in zip(reversed(matches), reversed(signs)):
            new_expr = new_expr[: match.start()] + sign + new_expr[match.end() :]
        results.append(new_expr)

    return results


# 複数数式ブロックのマージ: $expr1$ <区切り> $expr2$ → $expr1, expr2$
# parse() は最後の数式ブロックのみ拾うため、区切り文字で連結されたブロックを統合する
# 対応デリミタ: $...$, $$...$$, \[...\], \(...\)
# Note: $$ は $+$ に分割しないよう、$$ を先にマッチさせる
_MATH_BLOCK_SEP_RE = re.compile(
    r"(?:\$\$|(?<!\$)\$(?!\$)|\\\]|\\\))"  # 前ブロックの閉じデリミタ
    r"\s*"                              # 空白
    r"(?:"
    r"[.,、，;]"                          # 句読点・カンマ・セミコロン
    r"|または|かつ"                      # 日本語接続詞
    r"|\bor\b|\band\b"                   # 英語接続詞
    r"|\\(?:text|mathrm|textbf|textit)\s*\{\s*(?:または|かつ|or|and)\s*\}"
    r")?"                               # 区切りは省略可 (空白のみ)
    r"\s*"                              # 空白
    r"(?:\$\$|(?<!\$)\$(?!\$)|\\\[|\\\()",  # 次ブロックの開きデリミタ
    re.IGNORECASE,
)


def _merge_math_blocks(text: str) -> str:
    r"""隣接する数式ブロックを単一ブロックに統合する。

    $expr1$. $expr2$ → $expr1, expr2$
    \[expr1\]. \[expr2\] → \[expr1, expr2\]
    $expr1$ または $expr2$ → $expr1, expr2$

    対応デリミタ: $...$, \[...\], \(...\)
    parse() は複数ブロックの最後しか拾わないため、
    区切り文字で連結されたブロックを事前にマージする。
    """
    return _MATH_BLOCK_SEP_RE.sub(", ", text)


def _extended_parse(expr: str) -> list:
    r"""Extended parse with additional preprocessing.

    Features:
    - 確率変数の正規化: P(X=k) → p_{Xk}
    - 括弧サイズコマンド除去: \bigl, \bigr, \left, \right 等
    - LaTeX スペーシング除去: \,, \;, \quad 等 → スペース
    - 複数数式ブロックのマージ: $a$. $b$ → $a, b$
    - 不等号の正規化: \geqq → \geq, \leqq → \leq 等
    - 論理記号の正規化: \vee, \lor, \wedge, \land → , (カンマ)
    - \pm/\mp expansion: expands to both + and - variants, returns as FiniteSet
    - 改行除去: 数式ブロック内の改行を空白に置換
    - ベクトル記号正規化: 全バリアント → \vec{...}、単一文字のみ除去
    - 小数 → 分数 変換: 0.625 → \frac{625}{1000}
    """
    # 改行を空白に置換（$...\n...\n...$ のようなケースに対応）
    expr = expr.replace("\n", " ")
    # \varnothing → \emptyset 正規化（パーサが \varnothing を Symbol にしてしまうため）
    expr = expr.replace(r"\varnothing", r"\emptyset")
    expr = _normalize_prob_vars(expr)
    expr = _strip_vector_notation(expr)  # 単一文字は除去、複数文字は \vec{AB} のまま保持
    expr = _strip_bracket_sizing(expr)
    expr = _strip_latex_spacing(expr)
    expr = _merge_math_blocks(expr)
    expr = _normalize_inequalities(expr)
    expr = _normalize_logical_connectives(expr)
    expanded_exprs = _expand_pm_mp(expr)

    if len(expanded_exprs) == 1:
        result = parse(expr, extraction_config=_EXTRACTION_CONFIG)
        # Spurious EmptySet 修正:
        # (symbolic_fraction, symbolic_fraction) が Interval → EmptySet に誤解釈されるケースを
        # Tuple として再パースする
        if _is_spurious_emptyset(result, expr):
            tuple_result = _try_reparse_as_tuple(expr)
            if tuple_result is not None:
                return tuple_result
        return result

    all_values = set()
    all_equalities = []

    for expanded_expr in expanded_exprs:
        try:
            parsed = parse(expanded_expr, extraction_config=_EXTRACTION_CONFIG)
            if parsed:
                val = parsed[0]
                if isinstance(val, Eq):
                    all_equalities.append(val)
                elif hasattr(val, "__iter__") and not isinstance(val, str):
                    for v in val:
                        all_values.add(v)
                else:
                    all_values.add(val)
        except Exception:
            pass

    # 全展開結果が同じ LHS の Equality なら RHS の FiniteSet を統合
    # Note: parse が返す FiniteSet は latex2sympy2_extended.sets.FiniteSet であり、
    #       sympy.FiniteSet とは __eq__ が異なるため、L2SFiniteSet を使う必要がある
    if all_equalities and not all_values:
        lhs_set = {eq.lhs for eq in all_equalities}
        if len(lhs_set) == 1:
            lhs = lhs_set.pop()
            merged_rhs = set()
            for eq in all_equalities:
                rhs = eq.rhs
                if isinstance(rhs, FiniteSet):
                    merged_rhs.update(v.doit() for v in rhs)
                else:
                    merged_rhs.add(rhs.doit())
            return [Eq(lhs, L2SFiniteSet(*merged_rhs), evaluate=False)]

    if all_values:
        return [FiniteSet(*all_values)]
    return parse(expr, extraction_config=_EXTRACTION_CONFIG)


def _paren_variants(expr: str) -> list[str]:
    r"""括弧の種類を変えたバリエーションを生成する。

    (), [], \{\} を相互に置き換えた式を返す（元の式と同じものは除く）。
    \left/\right がある場合はそれを保持する。
    数式デリミタ \[, \], \(, \) は置換対象外。
    """
    s = expr
    has_brackets = False

    # 数式デリミタ \[, \], \(, \) を保護（括弧置換の対象外にする）
    s = s.replace(r"\[", "\x01DO\x01")
    s = s.replace(r"\]", "\x01DC\x01")
    s = s.replace(r"\(", "\x01IO\x01")
    s = s.replace(r"\)", "\x01IC\x01")

    # \sqrt[n] の [n] を保護（n乗根の記法を壊さない）
    import re as _re

    _sqrt_bracket_re = _re.compile(r"\\sqrt\[([^\]]*)\]")
    _sqrt_placeholders: list[str] = []
    def _protect_sqrt(m: _re.Match) -> str:
        idx = len(_sqrt_placeholders)
        _sqrt_placeholders.append(m.group(0))
        return f"\x02SQRT{idx}\x02"
    s = _sqrt_bracket_re.sub(_protect_sqrt, s)

    # \left/\right 付き括弧をプレースホルダに置換（長いパターンから先に）
    for pat in [r"\left\{", r"\left(", r"\left["]:
        if pat in s:
            s = s.replace(pat, "\x00LO\x00")
            has_brackets = True
    for pat in [r"\right\}", r"\right)", r"\right]"]:
        if pat in s:
            s = s.replace(pat, "\x00RC\x00")
            has_brackets = True

    # plain 括弧をプレースホルダに置換（\{ は ( より先に）
    for pat in [r"\{", "(", "["]:
        if pat in s:
            s = s.replace(pat, "\x00PO\x00")
            has_brackets = True
    for pat in [r"\}", ")", "]"]:
        if pat in s:
            s = s.replace(pat, "\x00PC\x00")
            has_brackets = True

    if not has_brackets:
        return []

    _BRACKET_TYPES = [
        (r"\left(", r"\right)", "(", ")"),
        (r"\left[", r"\right]", "[", "]"),
        (r"\left\{", r"\right\}", r"\{", r"\}"),
    ]

    variants = []
    for left_b, right_b, plain_l, plain_r in _BRACKET_TYPES:
        v = s
        v = v.replace("\x00LO\x00", left_b).replace("\x00RC\x00", right_b)
        v = v.replace("\x00PO\x00", plain_l).replace("\x00PC\x00", plain_r)
        # 保護した数式デリミタを復元
        v = v.replace("\x01DO\x01", r"\[").replace("\x01DC\x01", r"\]")
        v = v.replace("\x01IO\x01", r"\(").replace("\x01IC\x01", r"\)")
        # \sqrt[n] を復元
        for idx, orig in enumerate(_sqrt_placeholders):
            v = v.replace(f"\x02SQRT{idx}\x02", orig)
        if v != expr:
            variants.append(v)

    return variants


def _relational_to_interval(rel, var):
    """単一の不等式を Interval に変換する。

    var が不等式の片側に単独で出現する場合のみ対応。
    例: x < a → (-∞, a), b < x → (b, ∞)
    """
    from sympy import oo, StrictLessThan, StrictGreaterThan, LessThan, GreaterThan

    lhs, rhs = rel.lhs, rel.rhs
    var_on_left = (lhs == var)
    var_on_right = (rhs == var)
    if not var_on_left and not var_on_right:
        return None

    if isinstance(rel, StrictLessThan):
        # var < rhs or lhs < var
        return Interval.open(-oo, rhs) if var_on_left else Interval.open(lhs, oo)
    elif isinstance(rel, LessThan):
        return Interval(-oo, rhs) if var_on_left else Interval(lhs, oo)
    elif isinstance(rel, StrictGreaterThan):
        return Interval.open(rhs, oo) if var_on_left else Interval.open(-oo, lhs)
    elif isinstance(rel, GreaterThan):
        return Interval(rhs, oo) if var_on_left else Interval(-oo, lhs)
    return None


def _flatten_to_atoms(expr) -> list | None:
    """And/連鎖不等式を再帰展開してアトミックな Relational のリストを返す。

    Relational でも And でもない要素が含まれる場合は None を返す。
    """
    from sympy import And as SympyAnd

    if isinstance(expr, Relational):
        return [expr]
    if isinstance(expr, SympyAnd):
        atoms = []
        for arg in expr.args:
            sub = _flatten_to_atoms(arg)
            if sub is None:
                return None
            atoms.extend(sub)
        return atoms
    return None


def _normalize_ineq_direction(rel):
    """不等式を (simplify(lhs - rhs), strict?) に正規化する。

    全て > 0 (strict) または >= 0 (non-strict) 方向に統一し、
    simplify した差分式を返す。
    """
    from sympy import simplify as sym_simplify
    from sympy import StrictLessThan, LessThan, StrictGreaterThan, GreaterThan

    diff = rel.lhs - rel.rhs
    if isinstance(rel, (StrictLessThan, LessThan)):
        diff = -diff
    strict = isinstance(rel, (StrictLessThan, StrictGreaterThan))
    return (sym_simplify(diff), strict)


def _verify_inequality_atoms(g_val, p_val) -> bool:
    """FiniteSet{不等式} 同士をアトミック不等式の集合として比較する。

    And や連鎖不等式を展開し、各不等式を (lhs - rhs) > 0 方向に正規化。
    simplify 後の集合が一致すれば True。
    """
    if not isinstance(g_val, L2SFiniteSet) or not isinstance(p_val, L2SFiniteSet):
        return False

    # 全要素を展開してアトミック不等式のリストにする
    g_atoms = []
    for arg in g_val._unsorted_args:
        flat = _flatten_to_atoms(arg)
        if flat is None:
            return False
        g_atoms.extend(flat)

    p_atoms = []
    for arg in p_val._unsorted_args:
        flat = _flatten_to_atoms(arg)
        if flat is None:
            return False
        p_atoms.extend(flat)

    if len(g_atoms) != len(p_atoms):
        return False

    # 各不等式を正規化して集合比較
    try:
        g_normalized = {_normalize_ineq_direction(a) for a in g_atoms}
        p_normalized = {_normalize_ineq_direction(a) for a in p_atoms}
        return g_normalized == p_normalized
    except Exception:
        return False


def _inequalities_to_intervals(expr) -> list | None:
    """FiniteSet{不等式} の各不等式を Interval に変換したリストを返す。

    全要素が Relational (不等式) でない場合は None を返す。
    1. as_set() を試す（単変数の数値ケース）
    2. 失敗時は共通変数を特定し手動で区間構築（多変数・記号ケース）
    """
    if not isinstance(expr, L2SFiniteSet):
        return None

    args = list(expr._unsorted_args)

    # 全要素のアトムを収集
    all_atom_groups = []  # 各要素ごとのアトムリスト
    for arg in args:
        atoms = _flatten_to_atoms(arg)
        if atoms is None:
            return None
        all_atom_groups.append(atoms)

    # 方法1: 単変数の場合 as_set() を使う
    # 全アトムの自由変数が1つだけなら as_set() が正しく動作する
    all_free = set()
    for atoms in all_atom_groups:
        for a in atoms:
            all_free |= a.free_symbols
    if len(all_free) == 1:
        try:
            intervals = []
            for atoms in all_atom_groups:
                atom_intervals = [a.as_set() for a in atoms]
                if len(atom_intervals) == 1:
                    intervals.append(atom_intervals[0])
                else:
                    # And のアトムは交差
                    result = atom_intervals[0]
                    for iv in atom_intervals[1:]:
                        result = result.intersect(iv)
                    intervals.append(result)
            return intervals
        except Exception:
            pass

    # 方法2: 共通変数を特定して _relational_to_interval で手動変換
    common_vars = None
    for atoms in all_atom_groups:
        for a in atoms:
            syms = a.free_symbols
            common_vars = syms if common_vars is None else common_vars & syms
    if not common_vars:
        return None

    for var in sorted(common_vars, key=str):
        intervals = []
        ok = True
        for atoms in all_atom_groups:
            atom_intervals = []
            for a in atoms:
                iv = _relational_to_interval(a, var)
                if iv is None:
                    ok = False
                    break
                atom_intervals.append(iv)
            if not ok:
                break
            if len(atom_intervals) == 1:
                intervals.append(atom_intervals[0])
            else:
                result = atom_intervals[0]
                for iv in atom_intervals[1:]:
                    result = result.intersect(iv)
                intervals.append(result)
        if ok:
            return intervals
    return None


def _combine_intervals(intervals: list):
    r"""Interval リストから Union と Intersection の候補を返す。

    \vee (OR) → Union, \wedge (AND) → Intersection の両方を返し、
    verify でマッチするほうを採用する。
    """
    candidates = []
    # Union (OR 解釈)
    try:
        candidates.append(Union(*intervals))
    except Exception:
        pass
    # Intersection (AND 解釈): reduce で逐次的に交差
    # EmptySet は除外（矛盾する不等式同士が空集合で一致する誤判定を防ぐ）
    if len(intervals) >= 2:
        try:
            result = intervals[0]
            for iv in intervals[1:]:
                result = result.intersect(iv)
            if result is not S.EmptySet:
                candidates.append(result)
        except Exception:
            pass
    elif len(intervals) == 1:
        candidates.append(intervals[0])
    return candidates


def _verify_interval(parsed_gold: list, parsed_pred: list) -> bool:
    """不等式 ↔ 区間表記の比較。

    1. アトミック不等式の集合比較（多変数・式変形を含むケース）
    2. FiniteSet{不等式} → Union/Intersection(Interval) 変換（単変数ケース）
    """
    g_val = parsed_gold[0] if parsed_gold else None
    p_val = parsed_pred[0] if parsed_pred else None

    # アトミック不等式の集合比較（多変数の連立不等式に有効）
    if g_val is not None and p_val is not None:
        if _verify_inequality_atoms(g_val, p_val):
            return True

    # 区間変換フォールバック（単変数の不等式 ↔ 区間表記）
    g_intervals = _inequalities_to_intervals(g_val) if g_val is not None else None
    p_intervals = _inequalities_to_intervals(p_val) if p_val is not None else None

    if g_intervals is None and p_intervals is None:
        return False

    g_candidates = _combine_intervals(g_intervals) if g_intervals else [g_val]
    p_candidates = _combine_intervals(p_intervals) if p_intervals else [p_val]
    for g_cmp in g_candidates:
        for p_cmp in p_candidates:
            try:
                if verify([g_cmp], [p_cmp]):
                    return True
            except Exception:
                continue
    return False


def _try_reparse_as_tuple(expr: str) -> list | None:
    """EmptySet に誤パースされた式をタプルとして再パースする。

    (expr1, expr2) のようなカンマ区切りの式を個別にパースして Tuple にまとめる。
    """
    # 数式デリミタを除去して内部を取得
    inner = expr.strip()
    for prefix, suffix in [
        (r"\[", r"\]"),
        (r"\(", r"\)"),
        ("$$", "$$"),
        ("$", "$"),
    ]:
        if inner.startswith(prefix) and inner.endswith(suffix):
            inner = inner[len(prefix) : -len(suffix)].strip()
            break

    # \boxed{...} を除去
    boxed_m = re.match(r"\\boxed\s*\{(.*)\}\s*$", inner, re.DOTALL)
    if boxed_m:
        inner = boxed_m.group(1).strip()

    # Q(...) のような関数呼び出しの外側を除去
    func_m = re.match(r"[A-Za-z]+\s*\((.*)\)\s*$", inner, re.DOTALL)
    if func_m:
        inner = func_m.group(1).strip()
    else:
        # \left( ... \right) を除去
        paren_m = re.match(
            r"(?:\\left\s*)?[(\[]\s*(.*?)\s*(?:\\right\s*)?[)\]]\s*$",
            inner,
            re.DOTALL,
        )
        if paren_m:
            inner = paren_m.group(1).strip()

    if "," not in inner:
        return None

    # カンマで分割（ネストを考慮した簡易分割）
    parts = _split_top_level_commas(inner)
    if len(parts) < 2:
        return None

    parsed_parts = []
    for part in parts:
        part_expr = f"${part.strip()}$"
        try:
            p = parse(part_expr, extraction_config=_EXTRACTION_CONFIG)
            if p and not isinstance(p[0], str):
                parsed_parts.append(p[0])
            else:
                return None
        except Exception:
            return None

    return [STuple(*parsed_parts), inner]


def _split_top_level_commas(s: str) -> list[str]:
    """ネストされた括弧・ブレースを考慮してトップレベルのカンマで分割する。"""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return parts


def _is_spurious_emptyset(parsed: list, raw: str) -> bool:
    """パーサが誤って EmptySet を生成したかどうかを判定する。

    (a, a) のようなタプル/座標が Interval → EmptySet と誤解釈されるケースを検出。
    raw 文字列に明示的な空集合記号がなければ誤解釈とみなす。
    """
    if not parsed:
        return False
    if parsed[0] is not S.EmptySet:
        return False
    # 明示的な空集合記号が含まれていれば正当な EmptySet
    if re.search(r"\\(?:emptyset|varnothing|empty)\b", raw):
        return False
    return True


def _verify_soft(prediction: str, gold: str) -> bool:
    """Soft evaluation: allows calculation, with bracket variant fallback."""
    parsed_gold = _extended_parse(gold)
    parsed_pred = _extended_parse(prediction)

    if verify(parsed_gold, parsed_pred):
        return True

    # 行列ベクトル ↔ タプル同一視フォールバック
    if _verify_matrix_vector(parsed_pred, parsed_gold):
        return True

    # 不等式 → 区間変換フォールバック
    if _verify_interval(parsed_gold, parsed_pred):
        return True

    # 括弧フォールバック: prediction の括弧の種類を変えて再検証
    # P(X=k) を先に正規化して、_paren_variants が P の括弧を変換しないようにする
    pred_for_paren = _normalize_prob_vars(prediction)
    gold_for_paren = _normalize_prob_vars(gold)
    for variant in _paren_variants(pred_for_paren):
        try:
            if verify(parsed_gold, _extended_parse(variant)):
                return True
        except Exception:
            continue

    # gold 側の括弧も変えて再検証
    for variant in _paren_variants(gold_for_paren):
        try:
            if verify(_extended_parse(variant), parsed_pred):
                return True
        except Exception:
            continue

    # \log (底なし) ↔ \ln フォールバック
    if _LOG_NO_BASE_RE.search(prediction) or _LOG_NO_BASE_RE.search(gold):
        pred_ln = _LOG_NO_BASE_RE.sub(r"\\ln", prediction)
        gold_ln = _LOG_NO_BASE_RE.sub(r"\\ln", gold)
        if (pred_ln, gold_ln) != (prediction, gold):
            try:
                if verify(_extended_parse(gold_ln), _extended_parse(pred_ln)):
                    return True
            except Exception:
                pass

    # Dict 形式フォールバック: {k: v, ...} を正規化して再検証（gold・pred 両方）
    gold_dict = _parse_dict_notation(gold)
    pred_dict = _parse_dict_notation(prediction)
    if gold_dict is not None:
        for candidate in gold_dict:
            try:
                if verify(_extended_parse(candidate), parsed_pred):
                    return True
            except Exception:
                continue
    if pred_dict is not None:
        for candidate in pred_dict:
            try:
                if verify(parsed_gold, _extended_parse(candidate)):
                    return True
            except Exception:
                continue
    # 両方 dict の場合は正規化同士で比較
    if gold_dict is not None and pred_dict is not None:
        for g_cand in gold_dict:
            for p_cand in pred_dict:
                try:
                    if verify(_extended_parse(g_cand), _extended_parse(p_cand)):
                        return True
                except Exception:
                    continue

    # ベクトル記号フォールバック: gold のベクトルトークンを pred にも適用して再比較
    if _verify_vector_fallback(prediction, gold):
        return True

    # 度数法 → ラジアン変換フォールバック
    if _verify_degree_radian(prediction, gold):
        return True

    # 小数近似フォールバック: 片方が小数のとき桁数に合わせて丸めて比較
    if _verify_numeric_approx(parsed_pred, parsed_gold):
        return True

    # 小数 → 分数変換フォールバック: 0.625x → \frac{5}{8}x 等
    if _verify_decimal_fraction(prediction, gold):
        return True

    # 裸の数学キーワードフォールバック: pi → \pi, sin → \sin 等
    if _verify_bare_keywords(prediction, gold):
        return True

    return False


def _verify_bare_keywords(prediction: str, gold: str) -> bool:
    r"""裸の数学キーワードを LaTeX コマンドに変換して再比較するフォールバック。

    pi → \pi, sin → \sin, cos → \cos 等。
    まず通常パース（pi = p*i）で比較し、一致しなかった場合に適用される。
    """
    pred_converted = _bare_keywords_to_latex(prediction)
    gold_converted = _bare_keywords_to_latex(gold)
    # 変換が発生しなければスキップ
    if pred_converted == prediction and gold_converted == gold:
        return False
    try:
        parsed_p = _extended_parse(pred_converted)
        parsed_g = _extended_parse(gold_converted)
        return verify(parsed_p, parsed_g)
    except Exception:
        return False


def _verify_decimal_fraction(prediction: str, gold: str) -> bool:
    r"""小数を分数に変換して再比較するフォールバック。

    0.625x → \frac{625}{1000}x (= \frac{5}{8}x) のように変換し、
    \frac{} 内や \sin() 内の小数にも対応する。
    """
    pred_converted = _decimal_to_fraction(prediction)
    gold_converted = _decimal_to_fraction(gold)
    # 変換が発生しなければスキップ
    if pred_converted == prediction and gold_converted == gold:
        return False
    try:
        parsed_p = _extended_parse(pred_converted)
        parsed_g = _extended_parse(gold_converted)
        return verify(parsed_p, parsed_g)
    except Exception:
        return False


def _verify_strict(prediction: str, gold: str) -> bool:
    """Strict evaluation: no calculation allowed."""
    parsed_pred = _extended_parse(prediction)
    parsed_gold = _extended_parse(gold)

    try:
        if srepr(parsed_pred[0]) == srepr(parsed_gold[0]):
            return True
    except Exception:
        try:
            p_latex = prediction.strip().strip("$")
            q_latex = gold.strip().strip("$")
            p_sympy = latex2sympy(p_latex)
            q_sympy = latex2sympy(q_latex)
            if srepr(p_sympy) == srepr(q_sympy):
                return True
        except Exception as e:
            err_console.log(f"Error occurred while verifying without calculation: {e}")

    # 小数近似フォールバック
    if _verify_numeric_approx(parsed_pred, parsed_gold):
        return True

    # 小数 → 分数変換フォールバック
    if _verify_decimal_fraction(prediction, gold):
        return True

    # 裸の数学キーワードフォールバック
    if _verify_bare_keywords(prediction, gold):
        return True

    return False


def _verify_complex(prediction: str, gold: str) -> bool:
    """Complex evaluation: 変数 i を虚数単位 I として扱い比較する。

    処理:
    1. 式をパースして sympy オブジェクトを取得
    2. free_symbols 中の i を虚数単位 I に置換
    3. verify() で直接比較（LaTeX 再変換を挟まない）
    4. .equals() でシンボル仮定を統一して数値的等価性チェック
       (Euler 公式など verify() だけでは判定できない同値性を判定)
    5. パース失敗・判定失敗時は _verify_soft にフォールバック
    """

    def _subs_i(expr):
        """式中の自由変数 i を虚数単位 I に置換する。"""
        if not hasattr(expr, "free_symbols"):
            return expr
        i_syms = [s for s in expr.free_symbols if str(s) == "i"]
        if not i_syms:
            return expr
        return expr.subs(i_syms[0], I)

    try:
        p_parsed = _extended_parse(prediction)[0]
        g_parsed = _extended_parse(gold)[0]
    except Exception:
        return _verify_soft(prediction, gold)

    p_sub = _subs_i(p_parsed)
    g_sub = _subs_i(g_parsed)

    # 1. verify() で直接比較
    if verify([g_sub], [p_sub]):
        return True

    # 2. シンボル仮定を統一して .equals() で数値的等価性チェック
    #    パーサーがシンボルに異なる仮定 (real=True/False) を付けることがあるため、
    #    gold のシンボルを pred のシンボルに統一する
    try:
        p_sym_map = {str(s): s for s in p_sub.free_symbols} if hasattr(p_sub, "free_symbols") else {}
        g_unified = g_sub
        if hasattr(g_sub, "free_symbols"):
            for s in list(g_sub.free_symbols):
                if str(s) in p_sym_map and s != p_sym_map[str(s)]:
                    g_unified = g_unified.subs(s, p_sym_map[str(s)])
        if hasattr(p_sub, "equals") and p_sub.equals(g_unified):
            return True
    except Exception:
        pass

    # 3. _verify_soft にフォールバック
    #    (括弧フォールバック、Eq vs FiniteSet の構造不一致等を処理)
    return _verify_soft(prediction, gold)


def _verify_single(
    prediction: str, gold: str, evaluation_method: Optional[EvaluationMethod]
) -> bool:
    """単一の gold に対して prediction を検証する。"""
    if evaluation_method == "strict":
        return _verify_strict(prediction, gold)
    elif evaluation_method == "complex":
        return _verify_complex(prediction, gold)
    elif evaluation_method == "complex-strict":
        # complex で合っていて、かつ strict でも合っていることを確認
        return _verify_complex(prediction, gold) and _verify_strict(prediction, gold)
    elif evaluation_method == "soft" or evaluation_method is None:
        return _verify_soft(prediction, gold)
    else:
        raise ValueError(f"Unknown evaluation method: {evaluation_method}")


def _verify_with_delimiter_fallback(
    prediction: str, gold: str, evaluation_method: Optional[EvaluationMethod]
) -> bool:
    """通常の評価を行い、False の場合は prediction を $...$ で囲んで再評価する。

    prediction にデリミタがない場合、parse が数値抽出のみとなり
    LaTeX 式が正しくパースされないことがある。フォールバックとして
    $...$ で囲むことで LaTeX パースを試みる。
    """
    if _verify_single(prediction, gold, evaluation_method):
        return True

    wrapped_prediction = _ensure_math_delimiters(prediction)
    if wrapped_prediction == prediction:
        return False  # 既にデリミタありなので再評価不要

    return _verify_single(wrapped_prediction, gold, evaluation_method)


def parse_and_verify(
    prediction: str,
    gold: str | list[str],
    evaluation_method: Optional[EvaluationMethod] = None,
) -> bool:
    """Parse and verify the prediction against the gold answer.

    gold が文字列のリストの場合、いずれか1つが一致すれば正解と判定する。
    Note: Returns False if any error occurs during parsing or verification.
    """
    if evaluation_method is None:
        warnings.warn(
            "evaluation_method is None, defaulting to 'soft'\n⚠️  評価スクリプトにフラグによる条件分岐が追加されました。フラグを含む新しいテストデータを使用してください。",
            UserWarning,
            stacklevel=2,
        )

    # prediction の前処理: XML タグ除去、数値間スペース正規化
    prediction = _strip_xml_tags(prediction)
    prediction = _normalize_digit_spaces(prediction)

    try:
        if isinstance(gold, list):
            return any(_verify_with_delimiter_fallback(prediction, g, evaluation_method) for g in gold)
        return _verify_with_delimiter_fallback(prediction, gold, evaluation_method)
    except (NotImplementedError, ValueError):
        raise  # Re-raise for caller to handle
    except Exception as e:
        err_console.log(f"Error occurred while verifying: {e}")
        return False


def accuracy(results: list[bool]) -> float:
    """Calculate accuracy from a list of boolean results."""
    return sum(results) / len(results) if results else 0.0


@app.command()
def math_eval(
    prediction_file: Annotated[str, typer.Argument(help="Path to the prediction file")],
    gold_file: Annotated[str, typer.Argument(help="Path to the gold file")],
    output_file: Annotated[
        str, typer.Option("--output-file", "-o", help="Path to the output file")
    ] = None,
) -> None:
    """Evaluate predictions on mathematical reasoning tasks."""
    id_prediction_map = load_examples(prediction_file, PredictionExample)
    id_gold_map = load_examples(gold_file, GoldExample)

    id_result_map: dict[str, bool] = {}
    for id_ in id_gold_map:
        if id_ not in id_prediction_map:
            err_console.log(
                f"Missing prediction for example ID '{id_}'; counting as incorrect."
            )
            id_result_map[id_] = False
            continue
        prediction = id_prediction_map[id_]
        gold = id_gold_map[id_]
        id_result_map[id_] = parse_and_verify(
            prediction.output, gold.solution, gold.evaluation_method
        )

    overall_accuracy = accuracy(list(id_result_map.values()))
    category_result_map: dict[str, list[bool]] = {}
    for id_, result in id_result_map.items():
        category = id_gold_map[id_].category
        category_result_map.setdefault(category, []).append(result)
    category_accuracies = {
        category: accuracy(results) for category, results in category_result_map.items()
    }

    table = Table(title="Evaluation Results")
    table.add_column("Category", justify="left")
    table.add_column("Accuracy", justify="right")
    for category, acc in category_accuracies.items():
        table.add_row(category, f"{acc:.3f}")
    table.add_row("[bold]Overall[/bold]", f"[bold]{overall_accuracy:.3f}[/bold]")
    console.print(table)

    if output_file:
        with open(output_file, "wt", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "overall_accuracy": overall_accuracy,
                        "category_accuracies": category_accuracies,
                        "detailed_results": id_result_map,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        err_console.log(f"Results written to '{output_file}'.")
