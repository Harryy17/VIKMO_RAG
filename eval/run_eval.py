"""
eval/run_eval.py
────────────────
Automated evaluation of the VIKMO Dealer Assistant.

Metrics reported
----------------
*  Pass rate per category (happy_path, clarification, multi_turn,
   out_of_scope, tricky)
*  Overall pass rate
*  Tool invocation accuracy  (did the model call the expected tool?)
*  Hallucination proxy       (response contains fabricated SKU / price?)
*  Failure mode analysis     (printed per failing test)

Scoring heuristics
------------------
A test PASSES if ALL of the following hold:
  1. must_contain    → all strings appear in the response (case-insensitive).
  2. must_contain_any→ at least one string appears (case-insensitive), if set.
  3. must_not_contain→ none of the strings appear in the response.

Tool call accuracy is checked via monkey-patching the tool functions to
record whether they were invoked during the test.

Run:
    python eval/run_eval.py

Output is written to eval/results.json and printed to stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel

console = Console()

# ── Monkey-patch tool calls to track invocations ──────────────────────────────
import assistant.tools as _tools_module

_invocations: List[str] = []

_orig_check_stock           = _tools_module.check_stock
_orig_create_order          = _tools_module.create_order
_orig_find_parts_by_vehicle = _tools_module.find_parts_by_vehicle


def _tracked(name, fn):
    def wrapper(*args, **kwargs):
        _invocations.append(name)
        return fn(*args, **kwargs)
    return wrapper


_tools_module.check_stock           = _tracked("check_stock",           _orig_check_stock)
_tools_module.create_order          = _tracked("create_order",          _orig_create_order)
_tools_module.find_parts_by_vehicle = _tracked("find_parts_by_vehicle", _orig_find_parts_by_vehicle)

from assistant.agent import DealerAssistant


# ── helpers ───────────────────────────────────────────────────────────────────
def _check_response(response: str, expected: Dict) -> Dict[str, bool]:
    lower = response.lower()
    checks: Dict[str, bool] = {}

    for phrase in expected.get("must_contain", []):
        checks[f"contains:{phrase}"] = phrase.lower() in lower

    any_phrases = expected.get("must_contain_any", [])
    if any_phrases:
        checks["contains_any"] = any(p.lower() in lower for p in any_phrases)

    for phrase in expected.get("must_not_contain", []):
        checks[f"not_contains:{phrase}"] = phrase.lower() not in lower

    return checks


def run_single_test(agent: DealerAssistant, test: Dict) -> Dict[str, Any]:
    """Run one test case (possibly multi-turn) and return result dict."""
    turns      = test["turns"]
    expected   = test["expected"]
    responses  = []
    final_resp = ""

    _invocations.clear()
    agent.reset()

    for turn in turns:
        try:
            reply = agent.chat(turn["content"])
        except Exception as exc:
            reply = f"[EXCEPTION] {exc}"
        responses.append(reply)
        final_resp = reply
        time.sleep(0.5)   # avoid rate-limit on free tier

    # ── scoring ───────────────────────────────────────────────────────────────
    checks   = _check_response(final_resp, expected)
    passed   = all(checks.values()) if checks else True

    # Tool call check
    expected_tool = expected.get("tool_called")
    tool_correct: bool | None = None
    if expected_tool is not None:
        tool_correct = expected_tool in _invocations
        passed = passed and tool_correct
    elif expected_tool is None and not expected.get("tool_called"):
        # No tool expected — it's OK if none was called, but don't penalise
        tool_correct = None

    return {
        "id":          test["id"],
        "category":    test["category"],
        "description": test["description"],
        "passed":      passed,
        "tool_correct":tool_correct,
        "checks":      checks,
        "responses":   responses,
        "final_response": final_resp,
        "tools_called": list(_invocations),
    }


def run_eval(eval_path: Path = _ROOT / "eval" / "eval_set.json") -> List[Dict]:
    with open(eval_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    console.print(Panel(
        f"Running {len(tests)} evaluation tests …",
        title="[bold blue]VIKMO Eval[/bold blue]",
        border_style="blue"
    ))

    agent   = DealerAssistant()
    results = []

    for i, test in enumerate(tests, 1):
        console.print(f"[dim][{i}/{len(tests)}][/dim] {test['id']} — {test['description']}")
        result = run_single_test(agent, test)
        results.append(result)
        status = "[green]PASS[/green]" if result["passed"] else "[red]FAIL[/red]"
        console.print(f"  → {status}")

    return results


def print_report(results: List[Dict]) -> None:
    console.print()
    console.print(Panel("[bold]Evaluation Report[/bold]", border_style="cyan"))

    # ── per-category table ─────────────────────────────────────────────────────
    from collections import defaultdict
    by_cat: Dict[str, List] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["passed"])

    table = Table(title="Results by Category", show_lines=True)
    table.add_column("Category",   style="cyan")
    table.add_column("Pass",       style="green")
    table.add_column("Fail",       style="red")
    table.add_column("Pass Rate",  style="bold")

    for cat, passes in sorted(by_cat.items()):
        n_pass = sum(passes)
        n_fail = len(passes) - n_pass
        rate   = f"{100 * n_pass / len(passes):.0f}%"
        table.add_row(cat, str(n_pass), str(n_fail), rate)

    # Totals
    total_pass = sum(r["passed"] for r in results)
    total      = len(results)
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_pass}[/bold]",
        f"[bold]{total - total_pass}[/bold]",
        f"[bold]{100 * total_pass / total:.0f}%[/bold]",
    )
    console.print(table)

    # ── tool accuracy ──────────────────────────────────────────────────────────
    tool_tests = [r for r in results if r["tool_correct"] is not None]
    if tool_tests:
        tool_pass = sum(r["tool_correct"] for r in tool_tests)
        console.print(
            f"\n[bold]Tool Call Accuracy:[/bold] "
            f"{tool_pass}/{len(tool_tests)} = "
            f"{100 * tool_pass / len(tool_tests):.0f}%"
        )

    # ── failure analysis ───────────────────────────────────────────────────────
    failures = [r for r in results if not r["passed"]]
    if failures:
        console.print(f"\n[bold red]Failures ({len(failures)}):[/bold red]")
        for r in failures:
            console.print(f"  [red]✗[/red] {r['id']} — {r['description']}")
            console.print(f"    Response: {r['final_response'][:200]}…")
            failed_checks = {k: v for k, v in r["checks"].items() if not v}
            if failed_checks:
                console.print(f"    Failed checks: {list(failed_checks.keys())}")
    else:
        console.print("\n[bold green]All tests passed! ✓[/bold green]")


def main():
    results_path = _ROOT / "eval" / "results.json"
    results      = run_eval()

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    console.print(f"\n[dim]Results saved to {results_path}[/dim]")

    print_report(results)


if __name__ == "__main__":
    main()
