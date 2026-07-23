from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import sys

# Allow this script to be executed directly from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import fact_accuracy, mean


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            case = json.loads(line)
            case_id = str(case["case_id"])
            if case_id in cases:
                raise ValueError(f"Duplicate case_id {case_id!r} at line {line_number}")
            cases[case_id] = case
    return cases


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def update_fact_scores(
    rows: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
) -> None:
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in cases:
            raise KeyError(f"No case definition found for {case_id!r}")

        result = fact_accuracy(
            str(row.get("answer") or ""),
            cases[case_id].get("required_facts") or [],
        )
        row["fact_accuracy"] = (
            "" if result["fact_accuracy"] is None else str(result["fact_accuracy"])
        )
        row["facts_matched"] = str(result["facts_matched"])
        row["facts_total"] = str(result["facts_total"])
        row["fact_details"] = json.dumps(
            result["fact_details"],
            ensure_ascii=False,
        )


def write_rows(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_value(row: dict[str, Any]) -> float | None:
    value = str(row.get("fact_accuracy") or "").strip()
    return float(value) if value else None


def build_fact_updates(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_mode_updates: dict[str, Any] = {}

    for mode in sorted({str(row["mode"]) for row in rows}):
        subset = [row for row in rows if str(row["mode"]) == mode]
        values = [
            score
            for row in subset
            if (score := score_value(row)) is not None
        ]

        category_values: dict[str, list[float]] = defaultdict(list)
        for row in subset:
            score = score_value(row)
            if score is not None:
                category_values[str(row["expected_category"])].append(score)

        by_mode_updates[mode] = {
            "mean_fact_accuracy": mean(values),
            "fact_accuracy_by_category": {
                category: mean(scores)
                for category, scores in category_values.items()
            },
        }

    index = {
        (str(row["case_id"]), str(row["mode"])): row
        for row in rows
    }
    wins = ties = losses = 0
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        baseline = index.get((case_id, "baseline"))
        full = index.get((case_id, "full"))
        if baseline is None or full is None:
            continue
        baseline_score = score_value(baseline)
        full_score = score_value(full)
        if baseline_score is None or full_score is None:
            continue
        if full_score > baseline_score:
            wins += 1
        elif full_score < baseline_score:
            losses += 1
        else:
            ties += 1

    total = wins + ties + losses
    pairwise = {
        "full_vs_baseline_fact_accuracy": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate_excluding_ties": (
                wins / (wins + losses) if wins + losses else None
            ),
            "win_rate_including_ties_as_half": (
                (wins + 0.5 * ties) / total if total else None
            ),
        }
    }
    return by_mode_updates, pairwise


def update_summary(
    existing_summary_path: Path,
    output_summary_path: Path,
    by_mode_updates: dict[str, Any],
    pairwise: dict[str, Any],
) -> dict[str, Any]:
    if existing_summary_path.exists():
        with existing_summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {"by_mode": {}, "pairwise": {}}

    summary.setdefault("by_mode", {})
    for mode, updates in by_mode_updates.items():
        summary["by_mode"].setdefault(mode, {})
        summary["by_mode"][mode].update(updates)

    summary["pairwise"] = pairwise
    summary["rescored_at_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore existing answers without making new Gemini API calls."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--existing-summary",
        type=Path,
        default=Path("evaluation/results/conversation/summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "evaluation/results/conversation/per_case_results_rescored.csv"
        ),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("evaluation/results/conversation/summary_rescored.json"),
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    rows, fieldnames = load_rows(args.results)
    update_fact_scores(rows, cases)
    write_rows(rows, fieldnames, args.output_csv)

    by_mode_updates, pairwise = build_fact_updates(rows)
    summary = update_summary(
        args.existing_summary,
        args.output_summary,
        by_mode_updates,
        pairwise,
    )

    print(
        json.dumps(
            {
                "by_mode": by_mode_updates,
                "pairwise": pairwise,
                "output_csv": str(args.output_csv),
                "output_summary": str(args.output_summary),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()