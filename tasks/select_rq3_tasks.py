from __future__ import annotations

"""Select the formal RQ3 task set from the general Clean calibration output.

Default formal design: 40 tasks, balanced across the 10 task families whenever
possible (4/family).  A fixed middle-exposure anchor is derived from each task's
dominant Clean tool sequence so every RQ3 condition uses the same exposure site.
"""

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def middle_anchor_from_sequence(sequence: list[str]) -> dict[str, Any]:
    if not sequence:
        raise ValueError("Cannot choose an injection anchor from an empty sequence")
    n = len(sequence)
    scored: list[tuple[tuple[int, int, float, int], int]] = []
    for p in range(1, n + 1):
        norm = p / n
        scored.append(
            (
                (
                    0 if 0.30 <= norm <= 0.60 else 1,
                    0 if (n - p) >= 2 else 1,
                    abs(norm - 0.50),
                    p,
                ),
                p,
            )
        )
    _, p = min(scored, key=lambda x: x[0])
    tool = sequence[p - 1]
    occurrence = sum(1 for x in sequence[:p] if x == tool)
    return {
        "stage": "middle",
        "tool": tool,
        "occurrence": occurrence,
        "clean_dominant_position": p,
        "normalized_position": round(p / n, 6),
        "remaining_clean_actions": n - p,
        "source": "dominant_clean_sequence",
    }


def anchor_present_in_all_clean_runs(
    clean_rows: list[dict[str, Any]],
    task_id: str,
    anchor: dict[str, Any],
    expected_repeats: int = 5,
) -> bool:
    rows = [r for r in clean_rows if str(r.get("task_id")) == task_id]
    if len(rows) < expected_repeats:
        return False
    tool = str(anchor["tool"])
    occurrence = int(anchor["occurrence"])
    for row in rows[:expected_repeats]:
        seen = 0
        found = False
        for ev in row.get("trajectory") or []:
            if ev.get("tool") == tool:
                seen += 1
                if seen == occurrence:
                    found = True
                    break
        if not found:
            return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select PTTA RQ3 formal task set")
    p.add_argument("--selected-tasks", type=Path, required=True)
    p.add_argument("--calibration-results", type=Path, required=True)
    p.add_argument(
        "--clean-runs",
        type=Path,
        default=None,
        help="Defaults to clean_runs.jsonl beside calibration_results.jsonl.",
    )
    p.add_argument("--output", type=Path, default=Path("results/rq3/rq3_tasks.json"))
    p.add_argument("--report", type=Path, default=Path("results/rq3/rq3_selection_report.json"))
    p.add_argument("--total", type=int, default=40)
    p.add_argument("--per-family", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.selected_tasks)
    tasks = list(payload.get("tasks") or [])
    cal = {str(r["task_id"]): r for r in load_jsonl(args.calibration_results) if r.get("passed")}
    clean_path = args.clean_runs or (args.calibration_results.parent / "clean_runs.jsonl")
    clean_rows = load_jsonl(clean_path)

    eligible_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []

    for original in tasks:
        task_id = str(original["task_id"])
        row = cal.get(task_id)
        reasons: list[str] = []
        if row is None:
            reasons.append("missing_passed_calibration_record")
        if not bool((original.get("rq_eligibility") or {}).get("rq3", False)):
            reasons.append("rq3_not_eligible")
        if reasons:
            excluded.append({"task_id": task_id, "reasons": reasons})
            continue

        assert row is not None
        dominant = list(row.get("dominant_tool_sequence") or [])
        if not dominant:
            excluded.append({"task_id": task_id, "reasons": ["missing_dominant_sequence"]})
            continue

        task = copy.deepcopy(original)
        task["formal_clean_reference"] = {
            "clean_chain_length_median": row.get("clean_chain_length_median"),
            "dominant_tool_sequence": dominant,
            "dominant_structure_count": row.get("dominant_structure_count"),
            "median_clean_clean_structural_cdr": row.get("median_clean_clean_structural_cdr"),
        }
        task["formal_injection_target"] = middle_anchor_from_sequence(dominant)
        if not anchor_present_in_all_clean_runs(clean_rows, task_id, task["formal_injection_target"]):
            excluded.append({"task_id": task_id, "reasons": ["formal_middle_anchor_not_present_in_all_5_clean_repeats"]})
            continue
        eligible_by_family[str(task.get("family", "unknown"))].append(task)

    families = sorted(eligible_by_family)
    if args.per_family * len(families) != args.total:
        raise SystemExit(
            f"Formal balance mismatch: {len(families)} families x {args.per_family} "
            f"!= requested total {args.total}. Adjust explicitly."
        )

    shortages = {
        fam: args.per_family - len(rows)
        for fam, rows in eligible_by_family.items()
        if len(rows) < args.per_family
    }
    if shortages:
        report = {
            "status": "FAILED_INSUFFICIENT_FAMILY_CAPACITY",
            "available_by_family": {k: len(v) for k, v in sorted(eligible_by_family.items())},
            "shortages": shortages,
            "excluded": excluded,
        }
        save_json(args.report, report)
        raise SystemExit(f"RQ3 family balance cannot be satisfied. See {args.report}")

    final: list[dict[str, Any]] = []
    for fam in families:
        rows = sorted(
            eligible_by_family[fam],
            key=lambda t: (
                float((t.get("formal_clean_reference") or {}).get("median_clean_clean_structural_cdr") or 0.0),
                str(t["task_id"]),
            ),
        )
        final.extend(rows[: args.per_family])

    family_counts = Counter(str(t.get("family", "unknown")) for t in final)
    out = {
        "benchmark_name": "ptta_rq3_formal_tasks",
        "version": "1.0",
        "source_selected_tasks": str(args.selected_tasks),
        "source_calibration_results": str(args.calibration_results),
        "source_clean_runs": str(clean_path),
        "selection_rule": {
            "total": args.total,
            "per_family": args.per_family,
            "ranking_within_family": "lower Clean-Clean structural CDR, then task_id",
            "injection_anchor": "dominant Clean sequence position closest to normalized 0.5",
        },
        "task_count": len(final),
        "family_counts": dict(sorted(family_counts.items())),
        "tasks": final,
    }
    report = {
        "status": "OK",
        "available_by_family": {k: len(v) for k, v in sorted(eligible_by_family.items())},
        "selected_by_family": dict(sorted(family_counts.items())),
        "selected_task_ids": [t["task_id"] for t in final],
        "excluded": excluded,
    }
    save_json(args.output, out)
    save_json(args.report, report)
    print(f"Selected {len(final)} RQ3 tasks -> {args.output}")
    print(f"Family counts: {dict(sorted(family_counts.items()))}")


if __name__ == "__main__":
    main()
