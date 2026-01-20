import argparse
import ast
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def parse_function_name(input_str: str) -> str:
    try:
        expr = ast.parse(input_str, mode="eval").body
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name):
                return expr.func.id
            if isinstance(expr.func, ast.Attribute):
                return expr.func.attr
    except Exception:
        return "unknown"
    return "unknown"


def _classify_node(node: ast.AST) -> str:
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.Constant):
        return type(node.value).__name__
    return type(node).__name__


def parse_arg_types(input_str: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    try:
        expr = ast.parse(input_str, mode="eval").body
        if not isinstance(expr, ast.Call):
            return [], []
        arg_types = [_classify_node(arg) for arg in expr.args]
        kw_types = [(kw.arg, _classify_node(kw.value)) for kw in expr.keywords if kw.arg]
        return arg_types, kw_types
    except Exception:
        return [], []


def _limit_items(items, max_hist: int):
    if max_hist is None or max_hist < 0:
        return items
    return items[:max_hist]


def summarize(results: Dict[str, Any], num_examples_match: int, num_examples_partial: int, max_hist: int):
    preds = results.get("predictions", [])
    if not preds:
        print("No predictions found.")
        return

    matches = [p for p in preds if p.get("match") == 1]
    partials = [
        p for p in preds
        if 0.0 < float(p.get("char_accuracy", 0.0)) < 1.0
    ]
    acc = mean([p.get("match", 0) for p in preds])
    char_acc = mean([float(p.get("char_accuracy", 0.0)) for p in preds])

    print(f"Total: {len(preds)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Avg char accuracy: {char_acc:.4f}")
    print("")

    print(f"Examples with match=1 (showing up to {num_examples_match}):")
    for ex in matches[:num_examples_match]:
        print(f"- input: {ex.get('input', '')}")
        print(f"  gold: {ex.get('gold', '')}")
        print(f"  pred: {ex.get('prediction', '')}")
    print("")

    print(f"Examples with 0<char_accuracy<1 (showing up to {num_examples_partial}):")
    for ex in partials[:num_examples_partial]:
        print(f"- input: {ex.get('input', '')}")
        print(f"  gold: {ex.get('gold', '')}")
        print(f"  pred: {ex.get('prediction', '')}")
        print(f"  char_accuracy: {ex.get('char_accuracy', 0.0):.4f}")
    print("")

    fn_counts = Counter()
    fn_match_counts = Counter()

    for ex in preds:
        input_str = ex.get("input", "")
        fn = parse_function_name(input_str)
        fn_counts[fn] += 1
        if ex.get("match") == 1:
            fn_match_counts[fn] += 1

    print("Top functions by frequency:")
    for fn, count in _limit_items(fn_counts.most_common(), max_hist):
        print(f"- {fn}: {count}")
    print("")

    print("Top functions by match=1 frequency:")
    for fn, count in _limit_items(fn_match_counts.most_common(), max_hist):
        print(f"- {fn}: {count}")
    print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--num_examples_match", type=int, default=3)
    parser.add_argument("--num_examples_partial", type=int, default=3)
    parser.add_argument("--max_hist", type=int, default=-1)
    args = parser.parse_args()

    results = load_results(args.results)
    summarize(results, args.num_examples_match, args.num_examples_partial, args.max_hist)


if __name__ == "__main__":
    main()
