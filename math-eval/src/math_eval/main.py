"""Evaluate predictions on mathematical reasoning tasks.

Usage:
    uv run math-eval <prediction-file> <gold-file> [-o <output-file>]
"""

import json
import dataclasses
import warnings
from typing import Any, Literal
from typing import Optional

import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.table import Table

from math_verify import parse, verify
from latex2sympy2_extended import latex2sympy
from sympy import I, latex
from sympy import srepr

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
    evaluation_method: Optional[EvaluationMethod] = None


@dataclasses.dataclass
class GoldExample:
    id: int
    problem: str
    solution: str
    category: str
    unit: str
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


def _verify_soft(prediction: str, gold: str) -> bool:
    """Soft evaluation: allows calculation."""
    return verify(parse(gold), parse(prediction))


def _verify_strict(prediction: str, gold: str) -> bool:
    """Strict evaluation: no calculation allowed."""
    try:
        p = parse(prediction)[0]
        q = parse(gold)[0]
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
    """Complex evaluation: to be implemented."""

    def __convert_i_to_imag(expr_str: str) -> str:
        parsed_expr = parse(expr_str)[0]
        has_i = any(str(sym) == "i" for sym in parsed_expr.free_symbols)
        if has_i:
            i_symbol = [sym for sym in parsed_expr.free_symbols if str(sym) == "i"][0]
            sympy_expr = parsed_expr.subs(i_symbol, I).expand().simplify()
            latex_expr = latex(sympy_expr)
        else:
            latex_expr = expr_str
        return "$" + latex_expr + "$"

    prediction = __convert_i_to_imag(prediction)
    gold = __convert_i_to_imag(gold)
    return _verify_soft(prediction, gold)


def parse_and_verify(
    prediction: str, gold: str, evaluation_method: Optional[EvaluationMethod] = None
) -> bool:
    """Parse and verify the prediction against the gold answer.

    Note: Returns False if any error occurs during parsing or verification.
    """
    if evaluation_method is None:
        warnings.warn(
            "evaluation_method is None, defaulting to 'soft'\n⚠️  評価スクリプトにフラグによる条件分岐が追加されました。フラグを含む新しいテストデータを使用してください。",
            UserWarning,
            stacklevel=2,
        )

    try:
        if evaluation_method == "strict":
            return _verify_strict(prediction, gold)
        elif evaluation_method == "complex":
            return _verify_complex(prediction, gold)
        elif evaluation_method == "soft" or evaluation_method is None:
            return _verify_soft(prediction, gold)
        else:
            raise ValueError(f"Unknown evaluation method: {evaluation_method}")
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
