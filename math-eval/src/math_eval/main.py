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
from sympy import Eq, FiniteSet, I, Interval, Union, latex, srepr
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

EvaluationMethod = Literal["soft", "strict", "complex"]


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
_DIGIT_SPACE_RE = re.compile(r"(?<=[0-9])\s+(?=[0-9.])|(?<=[.])\s+(?=[0-9])")


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
    r"|\\text\s*\{(?:または|かつ|or|and)\}"
    r"|\b(?:or|and)\b"
)


def _normalize_logical_connectives(expr: str) -> str:
    r"""論理記号 (\vee, \lor, \text{または}, or 等) をカンマに正規化する。"""
    return _LOGICAL_CONNECTIVE_RE.sub(",", expr)


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
# 対応デリミタ: $...$, \[...\], \(...\)
_MATH_BLOCK_SEP_RE = re.compile(
    r"(?:\$|\\\]|\\\))"                 # 前ブロックの閉じデリミタ
    r"\s*"                              # 空白
    r"(?:"
    r"[.,、，;]"                          # 句読点・カンマ・セミコロン
    r"|または|かつ"                      # 日本語接続詞
    r"|\bor\b|\band\b"                   # 英語接続詞
    r")?"                               # 区切りは省略可 (空白のみ)
    r"\s*"                              # 空白
    r"(?:\$|\\\[|\\\()"                 # 次ブロックの開きデリミタ
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
    - 複数 $...$ ブロックのマージ
    - 不等号の正規化: \geqq → \geq, \leqq → \leq 等
    - 論理記号の正規化: \vee, \lor, \wedge, \land → , (カンマ)
    - \pm/\mp expansion: expands to both + and - variants, returns as FiniteSet
    """
    expr = _merge_math_blocks(expr)
    expr = _normalize_inequalities(expr)
    expr = _normalize_logical_connectives(expr)
    expanded_exprs = _expand_pm_mp(expr)

    if len(expanded_exprs) == 1:
        return parse(expr, extraction_config=_EXTRACTION_CONFIG)

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
    # And や連鎖不等式を含む場合もアトミック展開して Relational のみ抽出
    flat_args = []
    for arg in args:
        flat = _flatten_to_atoms(arg)
        if flat is None:
            return None
        flat_args.extend(flat)
    args = flat_args

    # 方法1: as_set() を試す
    intervals = []
    try:
        for arg in args:
            intervals.append(arg.as_set())
        return intervals
    except Exception:
        pass

    # 方法2: 共通変数を特定して手動変換
    # 全不等式に共通する変数を探し、各変数で変換を試みる
    common_vars = None
    for arg in args:
        syms = arg.free_symbols
        common_vars = syms if common_vars is None else common_vars & syms
    if not common_vars:
        return None

    for var in sorted(common_vars, key=str):
        intervals = []
        for arg in args:
            iv = _relational_to_interval(arg, var)
            if iv is None:
                break
            intervals.append(iv)
        else:
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
    if len(intervals) >= 2:
        try:
            result = intervals[0]
            for iv in intervals[1:]:
                result = result.intersect(iv)
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


def _verify_soft(prediction: str, gold: str) -> bool:
    """Soft evaluation: allows calculation, with bracket variant fallback."""
    parsed_gold = _extended_parse(gold)
    parsed_pred = _extended_parse(prediction)

    if verify(parsed_gold, parsed_pred):
        return True

    # 不等式 → 区間変換フォールバック
    if _verify_interval(parsed_gold, parsed_pred):
        return True

    # 括弧フォールバック: prediction の括弧の種類を変えて再検証
    for variant in _paren_variants(prediction):
        try:
            if verify(parsed_gold, _extended_parse(variant)):
                return True
        except Exception:
            continue

    # gold 側の括弧も変えて再検証
    for variant in _paren_variants(gold):
        try:
            if verify(_extended_parse(variant), parsed_pred):
                return True
        except Exception:
            continue

    return False


def _verify_strict(prediction: str, gold: str) -> bool:
    """Strict evaluation: no calculation allowed."""
    try:
        p = _extended_parse(prediction)[0]
        q = _extended_parse(gold)[0]
        return srepr(p) == srepr(q)
    except Exception:
        try:
            p_latex = prediction.strip().strip("$")
            q_latex = gold.strip().strip("$")
            p_sympy = latex2sympy(p_latex)
            q_sympy = latex2sympy(q_latex)
            return srepr(p_sympy) == srepr(q_sympy)
        except Exception as e:
            err_console.log(f"Error occurred while verifying without calculation: {e}")
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

    # prediction に含まれる XML タグ (<answer> 等) を除去し、数値間スペースを正規化
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
