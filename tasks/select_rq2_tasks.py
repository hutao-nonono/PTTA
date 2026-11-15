from __future__ import annotations

"""Select the formal RQ2 task set from the general Clean calibration output.

Inputs (from 01_calibrate_general_tasks.py):
  selected_tasks.json
  calibration_results.jsonl

Formal design:
  Short      2-4   clean tool calls
  Medium     5-7
  Long       8-10
  Very Long 11-12
  default quota = 36 tasks/group (144 total)

The selector uses the *observed median Clean trajectory length*, not the task's
hand-authored expected length.  It also fixes one reproducible middle-exposure
anchor per task from the dominant Clean tool sequence.  If a requested group is
under-filled, the script fails loudly by default instead of silently changing
the paper design.
"""

import argparse
import copy
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

GROUPS = {
    "Short": (2, 4),
    "Medium": (5, 7),
    "Long": (8, 10),
    "Very Long": (11, 12),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def length_group(length: float | int | None) -> str | None:
    if length is None:
        return None
    x = float(length)
    for name, (lo, hi) in GROUPS.items():
        if lo <= x <= hi:
            return name
    return None


def middle_anchor_from_sequence(sequence: list[str]) -> dict[str, Any]:
    """Choose a fixed semantic anchor closest to normalized 0.5.

    The injected observation belongs to the action at position p; therefore the
    first potentially affected action is p+1.  Prefer positions in [0.3, 0.6]
    and, when possible, leave >=2 downstream actions for r=2 Recovery.
    """
    if not sequence:
        raise ValueError("Cannot choose an injection anchor from an empty sequence")

    n = len(sequence)
    candidates: list[tuple[tuple[float, int, int], int]] = []
    for p in range(1, n + 1):
        norm = p / n
        in_middle = 0.30 <= norm <= 0.60
        leaves_two = (n - p) >= 2
        score = (
            0 if in_middle else 1,
            0 if leaves_two else 1,
            int(round(abs(norm - 0.50) * 1_000_000)),
        )
        candidates.append((score, p))

    _, position = min(candidates, key=lambda x: (x[0], x[1]))
    tool = sequence[position - 1]
    occurrence = sum(1 for x in sequence[:position] if x == tool)
    return {
        "stage": "middle",
        "tool": tool,
        "occurrence": occurrence,
        "clean_dominant_position": position,
        "normalized_position": round(position / n, 6),
        "remaining_clean_actions": n - position,
        "source": "dominant_clean_sequence",
    }


def balanced_take(tasks: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    """Deterministic round-robin across families to avoid one-family domination."""
    by_family: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for task in sorted(tasks, key=lambda t: str(t["task_id"])):
        by_family[str(task.get("family", "unknown"))].append(task)

    families = sorted(by_family)
    selected: list[dict[str, Any]] = []
    while len(selected) < quota:
        progressed = False
        for fam in families:
            if by_family[fam] and len(selected) < quota:
                selected.append(by_family[fam].popleft())
                progressed = True
        if not progressed:
            break
    return selected


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
    p = argparse.ArgumentParser(description="Select PTTA RQ2 formal task set")
    p.add_argument("--selected-tasks", type=Path, required=True)
    p.add_argument("--calibration-results", type=Path, required=True)
    p.add_argument(
        "--clean-runs",
        type=Path,
        default=None,
        help="Defaults to clean_runs.jsonl beside calibration_results.jsonl.",
    )
    p.add_argument("--output", type=Path, default=Path("results/rq2/rq2_tasks.json"))
    p.add_argument("--report", type=Path, default=Path("results/rq2/rq2_selection_report.json"))
    p.add_argument("--per-group", type=int, default=36)
    p.add_argument(
        "--allow-underfilled",
        action="store_true",
        help="Diagnostic only. Formal paper runs should normally omit this flag.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    selected_payload = load_json(args.selected_tasks)
    selected_tasks = list(selected_payload.get("tasks") or [])
    calibration_rows = load_jsonl(args.calibration_results)
    cal_by_id = {str(r["task_id"]): r for r in calibration_rows if r.get("passed")}
    clean_path = args.clean_runs or (args.calibration_results.parent / "clean_runs.jsonl")
    clean_rows = load_jsonl(clean_path)

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for original in selected_tasks:
        task_id = str(original["task_id"])
        cal = cal_by_id.get(task_id)
        reasons: list[str] = []
        if cal is None:
            reasons.append("missing_passed_calibration_record")
        if not bool((original.get("rq_eligibility") or {}).get("rq2", False)):
            reasons.append("rq2_not_eligible")

        if reasons:
            excluded.append({"task_id": task_id, "reasons": reasons})
            continue

        assert cal is not None
        group = length_group(cal.get("clean_chain_length_median"))
        if group is None:
            excluded.append(
                {
                    "task_id": task_id,
                    "reasons": ["observed_clean_length_outside_2_12"],
                    "clean_chain_length_median": cal.get("clean_chain_length_median"),
                }
            )
            continue

        dominant = list(cal.get("dominant_tool_sequence") or [])
        if not dominant:
            excluded.append({"task_id": task_id, "reasons": ["missing_dominant_sequence"]})
            continue

        task = copy.deepcopy(original)
        task["rq2_group"] = group
        task["formal_clean_reference"] = {
            "clean_chain_length_median": cal.get("clean_chain_length_median"),
            "clean_chain_length_min": cal.get("clean_chain_length_min"),
            "clean_chain_length_max": cal.get("clean_chain_length_max"),
            "dominant_tool_sequence": dominant,
            "dominant_structure_count": cal.get("dominant_structure_count"),
            "median_clean_clean_structural_cdr": cal.get("median_clean_clean_structural_cdr"),
        }
        task["formal_injection_target"] = middle_anchor_from_sequence(dominant)
        if not anchor_present_in_all_clean_runs(clean_rows, task_id, task["formal_injection_target"]):
            excluded.append({"task_id": task_id, "reasons": ["formal_middle_anchor_not_present_in_all_5_clean_repeats"]})
            continue
        eligible.append(task)

    by_group: dict[str, list[dict[str, Any]]] = {
        name: [t for t in eligible if t["rq2_group"] == name] for name in GROUPS
    }
    available = {name: len(rows) for name, rows in by_group.items()}
    shortages = {
        name: args.per_group - count
        for name, count in available.items()
        if count < args.per_group
    }

    if shortages and not args.allow_underfilled:
        report = {
            "status": "FAILED_INSUFFICIENT_GROUP_CAPACITY",
            "requested_per_group": args.per_group,
            "available_by_group": available,
            "shortages": shortages,
            "note": (
                "Do not silently change the RQ2 group definitions or quotas. "
                "Expand/rebalance the candidate pool or revise the manuscript design explicitly."
            ),
            "excluded": excluded,
        }
        save_json(args.report, report)
        raise SystemExit(
            "RQ2 selection cannot satisfy the formal 4-group quota. "
            f"Available={available}, shortages={shortages}. See {args.report}"
        )

    final: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for name in GROUPS:
        quota = min(args.per_group, len(by_group[name])) if args.allow_underfilled else args.per_group
        chosen = balanced_take(by_group[name], quota)
        final.extend(chosen)
        selected_counts[name] = len(chosen)

    final = sorted(final, key=lambda t: (list(GROUPS).index(t["rq2_group"]), str(t["task_id"])))
    family_counts = Counter(str(t.get("family", "unknown")) for t in final)

    output = {
        "benchmark_name": "ptta_rq2_formal_tasks",
        "version": "1.0",
        "source_selected_tasks": str(args.selected_tasks),
        "source_calibration_results": str(args.calibration_results),
        "source_clean_runs": str(clean_path),
        "selection_rule": {
            "length_source": "observed median Clean trajectory length",
            "groups": {k: list(v) for k, v in GROUPS.items()},
            "requested_per_group": args.per_group,
            "middle_anchor": "closest stable dominant-sequence position to normalized 0.5",
            "family_balancing": "deterministic round-robin within each length group",
        },
        "task_count": len(final),
        "group_counts": selected_counts,
        "family_counts": dict(sorted(family_counts.items())),
        "tasks": final,
    }
    report = {
        "status": "OK" if not shortages else "UNDERFILLED_DIAGNOSTIC",
        "available_by_group": available,
        "selected_by_group": selected_counts,
        "shortages": shortages,
        "excluded": excluded,
        "selected_task_ids": [t["task_id"] for t in final],
    }
    save_json(args.output, output)
    save_json(args.report, report)
    print(f"Selected {len(final)} RQ2 tasks -> {args.output}")
    print(f"Group counts: {selected_counts}")


if __name__ == "__main__":
    main()
