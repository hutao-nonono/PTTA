#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal RQ4 analysis for PTTA human-validation experiments.

This script replaces the earlier proxy/majority/simulated analysis pipeline.
It uses four independent sources:

1. annotator-1 labels;
2. annotator-2 labels;
3. blind adjudicated human labels (gold reference);
4. formal PTTA predictions.

It never creates simulated experimental values. Recovery-window and argument-
weight sensitivity results are read from the supplied sensitivity JSON file.

Main outputs
------------
- table1_rq4_interannotator_agreement.csv
- table2_rq4_ptta_vs_adjudicated.csv
- table3_rq4_fdp_accuracy.csv
- table4_rq4_recovery_denominators.csv
- table5_rq4_direction_distribution.csv
- table6_rq4_asr_baseline.csv
- table7_rq4_sensitivity_recovery_window.csv
- table8_rq4_sensitivity_arg_weight.csv
- table9_rq4_sampling_audit.csv
- supplementary_ptta_disagreements.csv
- supplementary_annotator_disagreements.csv
- bootstrap_core_metrics.csv
- validation_report.json
- rq4_summary.json
- figures/*.png

Notes on interpretation
-----------------------
* Adjudicated labels are the human reference. A1/A2 are used for IAA only.
* Recovery is evaluated separately for Structural and Parameter layers.
* Right-censored/N/A runs are NOT put into the stable-recovery denominator.
* Stable-recovery F1 is computed only on runs jointly eligible according to
  PTTA and the adjudicated human reference. Recovery-status agreement is
  reported separately so joint-eligibility filtering cannot hide censoring
  disagreements.
* FDP accuracy uses only pairs where both PTTA and adjudication define an FDP;
  missing-definition disagreements are reported explicitly.
* Strict ASR is evaluated only as an endpoint baseline for process deviation;
  it is not treated as a trajectory detector.
* Bootstrap confidence intervals are task-cluster bootstraps when task metadata
  is available from annotation_pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{lineno}: {exc}") from exc
    return rows


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_by_pair(rows: Sequence[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        pid = r.get("pair_id")
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"{name}: row without valid pair_id")
        if pid in out:
            raise ValueError(f"{name}: duplicate pair_id {pid}")
        out[pid] = r
    return out


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


ALLOWED_DIRECTIONS = {"attacker_aligned", "defensive", "incidental", "none"}
ALLOWED_RECOVERY_STATUS = {"eligible", "right_censored", "N/A"}


def validate_label_record(record: dict[str, Any], source: str) -> list[str]:
    """Check internal label logic without re-adjudicating human judgments."""
    issues: list[str] = []
    pid = record.get("pair_id", "<unknown>")

    s = record.get("structural_deviation")
    p = record.get("parameter_deviation")
    if not isinstance(s, bool):
        issues.append(f"{source}:{pid}: structural_deviation is not bool")
    if not isinstance(p, bool):
        issues.append(f"{source}:{pid}: parameter_deviation is not bool")
    if not isinstance(s, bool) or not isinstance(p, bool):
        return issues

    any_dev = s or p
    fdp = record.get("first_deviation_position")
    direction = record.get("direction")

    if any_dev and fdp is None:
        issues.append(f"{source}:{pid}: deviation=True but FDP is null")
    if not any_dev and fdp is not None:
        issues.append(f"{source}:{pid}: no deviation but FDP is non-null")
    if fdp is not None and (not isinstance(fdp, int) or fdp < 1):
        issues.append(f"{source}:{pid}: invalid FDP={fdp!r}")

    if direction not in ALLOWED_DIRECTIONS:
        issues.append(f"{source}:{pid}: invalid direction={direction!r}")
    elif any_dev and direction == "none":
        issues.append(f"{source}:{pid}: deviation=True but direction=none")
    elif not any_dev and direction != "none":
        issues.append(f"{source}:{pid}: no deviation but direction={direction}")

    for layer, dev in (("structural", s), ("parameter", p)):
        status_key = f"{layer}_recovery_status"
        recovery_key = f"{layer}_stable_recovery"
        status = record.get(status_key)
        recovery = record.get(recovery_key)

        if status not in ALLOWED_RECOVERY_STATUS:
            issues.append(f"{source}:{pid}: invalid {status_key}={status!r}")
            continue

        if not dev:
            if status != "N/A":
                issues.append(f"{source}:{pid}: {layer} dev=False but status={status}")
            if recovery is not None:
                issues.append(f"{source}:{pid}: {layer} dev=False but stable_recovery non-null")
        else:
            if status == "N/A":
                issues.append(f"{source}:{pid}: {layer} dev=True but status=N/A")
            if status == "eligible" and not isinstance(recovery, bool):
                issues.append(f"{source}:{pid}: {layer} eligible but recovery is not bool")
            if status == "right_censored" and recovery is not None:
                issues.append(f"{source}:{pid}: {layer} right_censored but recovery non-null")

    conf = record.get("confidence")
    if source in {"annotator_1", "annotator_2", "adjudicated"}:
        if not isinstance(conf, int) or not (1 <= conf <= 5):
            issues.append(f"{source}:{pid}: confidence must be integer 1..5")

    return issues


def validate_blind_input(rows: Sequence[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    forbidden_top_level = {"ptta", "strict_asr", "attack_success"}
    for r in rows:
        pid = r.get("pair_id", "<unknown>")
        present = sorted(k for k in forbidden_top_level if k in r)
        if present:
            issues.append(f"blind_input:{pid}: forbidden fields present: {present}")
    return issues


def compare_pair_sets(named_indices: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    issues: list[str] = []
    names = list(named_indices)
    if not names:
        return issues
    ref_name = names[0]
    ref = set(named_indices[ref_name])
    for name in names[1:]:
        cur = set(named_indices[name])
        missing = sorted(ref - cur)
        extra = sorted(cur - ref)
        if missing:
            issues.append(f"{name}: missing {len(missing)} pair_ids relative to {ref_name}")
        if extra:
            issues.append(f"{name}: extra {len(extra)} pair_ids relative to {ref_name}")
    return issues


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _as_object_array(x: Sequence[Any]) -> np.ndarray:
    return np.asarray(list(x), dtype=object)


def raw_agreement(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if len(y_true) == 0:
        return float("nan")
    a = _as_object_array(y_true)
    b = _as_object_array(y_pred)
    return float(np.mean(a == b))


def cohens_kappa(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if len(y_true) == 0:
        return float("nan")
    a = _as_object_array(y_true)
    b = _as_object_array(y_pred)
    classes = sorted(set(a.tolist()) | set(b.tolist()), key=lambda z: str(z))
    po = float(np.mean(a == b))
    pe = 0.0
    for c in classes:
        pe += float(np.mean(a == c) * np.mean(b == c))
    if math.isclose(pe, 1.0):
        return 1.0 if math.isclose(po, 1.0) else float("nan")
    return (po - pe) / (1.0 - pe)


def binary_scores(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    if len(yt) == 0:
        return {k: float("nan") for k in ("accuracy", "precision", "recall", "f1", "kappa")}
    tp = int(np.sum(yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    if math.isfinite(precision) and math.isfinite(recall) and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    elif tp == 0 and fp == 0 and fn == 0:
        f1 = 1.0
    else:
        f1 = 0.0
    return {
        "accuracy": (tp + tn) / len(yt),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": cohens_kappa(yt.tolist(), yp.tolist()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if len(y_true) == 0:
        return float("nan")
    classes = sorted(set(y_true) | set(y_pred), key=lambda z: str(z))
    vals: list[float] = []
    for c in classes:
        yt = [x == c for x in y_true]
        yp = [x == c for x in y_pred]
        vals.append(binary_scores(yt, yp)["f1"])
    return float(np.nanmean(vals))


def multiclass_scores(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float]:
    return {
        "accuracy": raw_agreement(y_true, y_pred),
        "kappa": cohens_kappa(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }


def recovery_outcome(r: dict[str, Any], layer: str) -> str:
    status = r[f"{layer}_recovery_status"]
    if status == "N/A":
        return "N/A"
    if status == "right_censored":
        return "right_censored"
    return "recovered" if r[f"{layer}_stable_recovery"] else "not_recovered"


def finite_or_none(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(float(x)) else None


def round_or_none(x: Any, digits: int = 4) -> Any:
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return x
    if not math.isfinite(xf):
        return None
    return round(xf, digits)


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    estimate: float
    ci_low: float | None
    ci_high: float | None
    valid_resamples: int


def cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    metric_fn: Callable[[Sequence[dict[str, Any]]], float],
    *,
    cluster_key: str,
    B: int,
    seed: int,
) -> BootstrapResult:
    est = metric_fn(rows)
    if B <= 0 or not rows:
        return BootstrapResult(est, None, None, 0)

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r[cluster_key]].append(r)
    clusters = list(groups)
    if len(clusters) < 2:
        return BootstrapResult(est, None, None, 0)

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(B):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot: list[dict[str, Any]] = []
        for c in sampled:
            boot.extend(groups[c])
        v = metric_fn(boot)
        if v is not None and math.isfinite(float(v)):
            vals.append(float(v))
    if not vals:
        return BootstrapResult(est, None, None, 0)
    lo, hi = np.quantile(np.asarray(vals), [0.025, 0.975])
    return BootstrapResult(est, float(lo), float(hi), len(vals))


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def build_merged_rows(
    pair_ids: Sequence[str],
    a1: dict[str, dict[str, Any]],
    a2: dict[str, dict[str, Any]],
    adj: dict[str, dict[str, Any]],
    ptta: dict[str, dict[str, Any]],
    pair_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pid in pair_ids:
        meta = pair_meta[pid]
        cluster = f"{meta.get('source_rq','NA')}::{meta.get('task_id','NA')}"
        out.append({
            "pair_id": pid,
            "cluster_id": cluster,
            "source_rq": meta.get("source_rq"),
            "task_id": meta.get("task_id"),
            "repeat_id": meta.get("repeat_id"),
            "length_group": meta.get("length_group"),
            "template_family": meta.get("template_family"),
            "workflow_family": meta.get("workflow_family"),
            "a1": a1[pid],
            "a2": a2[pid],
            "adj": adj[pid],
            "ptta": ptta[pid],
        })
    return out


def rows_joint_eligible(rows: Sequence[dict[str, Any]], layer: str, left: str, right: str) -> list[dict[str, Any]]:
    key = f"{layer}_recovery_status"
    return [r for r in rows if r[left][key] == "eligible" and r[right][key] == "eligible"]


def agreement_metric(rows: Sequence[dict[str, Any]], left: str, right: str, field: str) -> tuple[int, float, float]:
    y1 = [r[left][field] for r in rows]
    y2 = [r[right][field] for r in rows]
    return len(rows), raw_agreement(y1, y2), cohens_kappa(y1, y2)


def confusion_matrix_labels(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(
    out_dir: Path,
    iaa_df: pd.DataFrame,
    ptta_df: pd.DataFrame,
    direction_true: Sequence[str],
    direction_pred: Sequence[str],
    recovery_df: pd.DataFrame,
    sens_recovery_df: pd.DataFrame,
) -> None:
    if plt is None:
        return
    figdir = out_dir / "figures"
    figdir.mkdir(exist_ok=True)

    # 1. Kappa overview
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    plot_rows = []
    for _, r in iaa_df.iterrows():
        if pd.notna(r.get("kappa")):
            plot_rows.append((f"IAA: {r['metric']}", float(r["kappa"])))
    for _, r in ptta_df.iterrows():
        if pd.notna(r.get("kappa")):
            plot_rows.append((f"PTTA: {r['metric']}", float(r["kappa"])))
    x = np.arange(len(plot_rows))
    ax.bar(x, [v for _, v in plot_rows])
    ax.set_xticks(x, [k for k, _ in plot_rows], rotation=35, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Cohen's kappa")
    ax.set_title("RQ4 agreement with human annotation")
    fig.tight_layout()
    fig.savefig(figdir / "figure1_rq4_agreement_kappa.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Direction confusion matrix
    labels = ["attacker_aligned", "defensive", "incidental", "none"]
    cm = confusion_matrix_labels(direction_true, direction_pred, labels)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("PTTA")
    ax.set_ylabel("Adjudicated human label")
    ax.set_title("Direction-label confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figdir / "figure2_rq4_direction_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Recovery rates by source/layer
    sub = recovery_df[recovery_df["source"].isin(["Adjudicated", "PTTA"])].copy()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    labels_x = ["Adjudicated\nStructural", "PTTA\nStructural", "Adjudicated\nParameter", "PTTA\nParameter"]
    vals = []
    for source, layer in [("Adjudicated", "Structural"), ("PTTA", "Structural"), ("Adjudicated", "Parameter"), ("PTTA", "Parameter")]:
        hit = sub[(sub["source"] == source) & (sub["layer"] == layer)]
        vals.append(float(hit.iloc[0]["recovery_rate"]) if not hit.empty else np.nan)
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)), labels_x)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recovery | eligible")
    ax.set_title("Layer-specific stable recovery")
    fig.tight_layout()
    fig.savefig(figdir / "figure3_rq4_recovery_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Recovery-window sensitivity from supplied source file
    if not sens_recovery_df.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        ax.plot(sens_recovery_df["recovery_window"], sens_recovery_df["recovery_rate"], marker="o")
        ax.set_xticks(sorted(sens_recovery_df["recovery_window"].unique()))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recovery window r")
        ax.set_ylabel("Recovery rate")
        ax.set_title("Supplied recovery-window sensitivity")
        fig.tight_layout()
        fig.savefig(figdir / "figure4_rq4_recovery_window_sensitivity.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Formal RQ4 analysis for PTTA human validation")
    ap.add_argument("--annotator1", default="rq4_annotator1_labels.jsonl")
    ap.add_argument("--annotator2", default="rq4_annotator2_labels.jsonl")
    ap.add_argument("--adjudicated", default="rq4_adjudicated_labels.jsonl")
    ap.add_argument("--ptta", default="rq4_ptta_predictions.jsonl")
    ap.add_argument("--pairs", default="rq4_annotation_pairs.jsonl")
    ap.add_argument("--blind-input", default="rq4_annotation_input_blind.jsonl")
    ap.add_argument("--manifest", default="rq4_sampling_manifest.json")
    ap.add_argument("--schema", default="rq4_annotation_schema.json")
    ap.add_argument("--sensitivity-results", default="rq4_sensitivity_results.json")
    ap.add_argument("--sensitivity-grid", default="rq4_sensitivity_grid.json")
    ap.add_argument("--output", default="rq4_analysis_formal")
    ap.add_argument("--bootstrap", type=int, default=2000, help="task-cluster bootstrap resamples")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--allow-validation-warnings", action="store_true", help="do not abort on validation errors")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "annotator_1": Path(args.annotator1),
        "annotator_2": Path(args.annotator2),
        "adjudicated": Path(args.adjudicated),
        "ptta": Path(args.ptta),
        "pairs": Path(args.pairs),
        "blind_input": Path(args.blind_input),
        "manifest": Path(args.manifest),
        "schema": Path(args.schema),
        "sensitivity_results": Path(args.sensitivity_results),
        "sensitivity_grid": Path(args.sensitivity_grid),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    a1_rows = load_jsonl(paths["annotator_1"])
    a2_rows = load_jsonl(paths["annotator_2"])
    adj_rows = load_jsonl(paths["adjudicated"])
    ptta_rows = load_jsonl(paths["ptta"])
    pair_rows = load_jsonl(paths["pairs"])
    blind_rows = load_jsonl(paths["blind_input"])
    manifest = load_json(paths["manifest"])
    schema = load_json(paths["schema"])
    sensitivity = load_json(paths["sensitivity_results"])
    sensitivity_grid = load_json(paths["sensitivity_grid"])

    a1 = index_by_pair(a1_rows, "annotator_1")
    a2 = index_by_pair(a2_rows, "annotator_2")
    adj = index_by_pair(adj_rows, "adjudicated")
    ptta = index_by_pair(ptta_rows, "ptta")
    pair_meta = index_by_pair(pair_rows, "pairs")
    blind = index_by_pair(blind_rows, "blind_input")

    # ------------------------------- validation ----------------------------
    validation_issues: list[str] = []
    validation_issues.extend(compare_pair_sets({
        "annotator_1": a1,
        "annotator_2": a2,
        "adjudicated": adj,
        "ptta": ptta,
        "pairs": pair_meta,
        "blind_input": blind,
    }))
    for source, rows in (("annotator_1", a1_rows), ("annotator_2", a2_rows), ("adjudicated", adj_rows), ("ptta", ptta_rows)):
        for rec in rows:
            validation_issues.extend(validate_label_record(rec, source))
    validation_issues.extend(validate_blind_input(blind_rows))

    total_pairs = int(manifest.get("total_pairs", len(adj_rows)))
    if len(adj_rows) != total_pairs:
        validation_issues.append(f"manifest total_pairs={total_pairs} but adjudicated has {len(adj_rows)}")
    strict_asr_observed = sum(bool(r.get("strict_asr")) for r in ptta_rows)
    expected_success = manifest.get("stratification", {}).get("by_success", {}).get("true")
    if expected_success is not None and strict_asr_observed != int(expected_success):
        validation_issues.append(f"strict ASR count mismatch: PTTA={strict_asr_observed}, manifest={expected_success}")

    if int(schema.get("recovery_window", 2)) != 2:
        validation_issues.append("schema recovery_window is not the expected primary r=2")

    # Sensitivity scope audit: current results expose only the supplied sweeps.
    sens_weight_keys = set(str(k) for k in sensitivity.get("alignment_weights", {}).keys())
    configured_arg = set(str(x) for x in sensitivity_grid.get("alignment_weights", {}).get("arg_change", []))
    sensitivity_notes: list[str] = []
    if sens_weight_keys and sens_weight_keys == configured_arg:
        sensitivity_notes.append("Sensitivity results contain the arg_change weight sweep; other configured alignment weights are not present in the results file.")

    validation_report = {
        "status": "PASS" if not validation_issues else "FAIL",
        "n_pairs": len(adj_rows),
        "strict_asr_successes": strict_asr_observed,
        "validation_issue_count": len(validation_issues),
        "issues": validation_issues,
        "notes": sensitivity_notes,
    }
    write_json(out / "validation_report.json", validation_report)
    if validation_issues and not args.allow_validation_warnings:
        preview = "\n".join(validation_issues[:20])
        raise RuntimeError(f"RQ4 validation failed with {len(validation_issues)} issue(s):\n{preview}")

    pair_ids = sorted(adj)
    merged = build_merged_rows(pair_ids, a1, a2, adj, ptta, pair_meta)

    # -------------------------- Table 1: IAA -------------------------------
    iaa_rows: list[dict[str, Any]] = []
    for metric, field in [
        ("Structural Deviation", "structural_deviation"),
        ("Parameter Deviation", "parameter_deviation"),
        ("Direction", "direction"),
        ("Structural Recovery Status", "structural_recovery_status"),
        ("Parameter Recovery Status", "parameter_recovery_status"),
    ]:
        n, acc, kap = agreement_metric(merged, "a1", "a2", field)
        iaa_rows.append({"metric": metric, "n": n, "raw_agreement": acc, "kappa": kap})

    for layer, label in (("structural", "Structural Stable Recovery"), ("parameter", "Parameter Stable Recovery")):
        sub = rows_joint_eligible(merged, layer, "a1", "a2")
        field = f"{layer}_stable_recovery"
        n, acc, kap = agreement_metric(sub, "a1", "a2", field)
        iaa_rows.append({"metric": label + " (joint eligible)", "n": n, "raw_agreement": acc, "kappa": kap})

    iaa_df = pd.DataFrame(iaa_rows)
    iaa_df = iaa_df.round({"raw_agreement": 4, "kappa": 4})
    iaa_df.to_csv(out / "table1_rq4_interannotator_agreement.csv", index=False)

    # -------------------- Table 2: PTTA vs adjudicated ---------------------
    ptta_eval_rows: list[dict[str, Any]] = []

    for label, field in (("Structural Deviation", "structural_deviation"), ("Parameter Deviation", "parameter_deviation")):
        yt = [bool(r["adj"][field]) for r in merged]
        yp = [bool(r["ptta"][field]) for r in merged]
        sc = binary_scores(yt, yp)
        ptta_eval_rows.append({
            "metric": label,
            "n": len(merged),
            "accuracy": sc["accuracy"],
            "kappa": sc["kappa"],
            "precision": sc["precision"],
            "recall": sc["recall"],
            "f1": sc["f1"],
            "macro_f1": np.nan,
        })

    yt_dir = [r["adj"]["direction"] for r in merged]
    yp_dir = [r["ptta"]["direction"] for r in merged]
    dsc = multiclass_scores(yt_dir, yp_dir)
    ptta_eval_rows.append({
        "metric": "Direction",
        "n": len(merged),
        "accuracy": dsc["accuracy"],
        "kappa": dsc["kappa"],
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "macro_f1": dsc["macro_f1"],
    })

    for layer, label in (("structural", "Structural"), ("parameter", "Parameter")):
        status_field = f"{layer}_recovery_status"
        yts = [r["adj"][status_field] for r in merged]
        yps = [r["ptta"][status_field] for r in merged]
        ssc = multiclass_scores(yts, yps)
        ptta_eval_rows.append({
            "metric": f"{label} Recovery Status",
            "n": len(merged),
            "accuracy": ssc["accuracy"],
            "kappa": ssc["kappa"],
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "macro_f1": ssc["macro_f1"],
        })

        sub = rows_joint_eligible(merged, layer, "adj", "ptta")
        rec_field = f"{layer}_stable_recovery"
        yt = [bool(r["adj"][rec_field]) for r in sub]
        yp = [bool(r["ptta"][rec_field]) for r in sub]
        rsc = binary_scores(yt, yp)
        ptta_eval_rows.append({
            "metric": f"{label} Stable Recovery (joint eligible)",
            "n": len(sub),
            "accuracy": rsc["accuracy"],
            "kappa": rsc["kappa"],
            "precision": rsc["precision"],
            "recall": rsc["recall"],
            "f1": rsc["f1"],
            "macro_f1": np.nan,
        })

    ptta_df = pd.DataFrame(ptta_eval_rows)
    ptta_df.to_csv(out / "table2_rq4_ptta_vs_adjudicated.csv", index=False, float_format="%.4f")

    # ---------------------------- Table 3: FDP -----------------------------
    gold_defined = [r for r in merged if r["adj"].get("first_deviation_position") is not None]
    pred_defined = [r for r in merged if r["ptta"].get("first_deviation_position") is not None]
    both_defined = [r for r in merged if r["adj"].get("first_deviation_position") is not None and r["ptta"].get("first_deviation_position") is not None]
    missing_disagreement = sum((r["adj"].get("first_deviation_position") is None) != (r["ptta"].get("first_deviation_position") is None) for r in merged)
    if both_defined:
        errors = np.asarray([
            abs(int(r["adj"]["first_deviation_position"]) - int(r["ptta"]["first_deviation_position"]))
            for r in both_defined
        ], dtype=float)
        fdp_row = {
            "n_total": len(merged),
            "n_adjudicated_defined": len(gold_defined),
            "n_ptta_defined": len(pred_defined),
            "n_both_defined": len(both_defined),
            "definition_mismatch_count": missing_disagreement,
            "exact_match_rate": float(np.mean(errors == 0)),
            "within_1_rate": float(np.mean(errors <= 1)),
            "mae": float(np.mean(errors)),
            "median_absolute_error": float(np.median(errors)),
        }
    else:
        fdp_row = {"n_total": len(merged), "n_adjudicated_defined": len(gold_defined), "n_ptta_defined": len(pred_defined), "n_both_defined": 0, "definition_mismatch_count": missing_disagreement, "exact_match_rate": np.nan, "within_1_rate": np.nan, "mae": np.nan, "median_absolute_error": np.nan}
    pd.DataFrame([fdp_row]).to_csv(out / "table3_rq4_fdp_accuracy.csv", index=False, float_format="%.4f")

    # ---------------------- Table 4: Recovery denominators -----------------
    rec_rows: list[dict[str, Any]] = []
    for source_key, source_name in (("a1", "Annotator 1"), ("a2", "Annotator 2"), ("adj", "Adjudicated"), ("ptta", "PTTA")):
        for layer, layer_name in (("structural", "Structural"), ("parameter", "Parameter")):
            dev_key = f"{layer}_deviation"
            status_key = f"{layer}_recovery_status"
            rec_key = f"{layer}_stable_recovery"
            deviations = sum(bool(r[source_key][dev_key]) for r in merged)
            eligible = sum(r[source_key][status_key] == "eligible" for r in merged)
            censored = sum(r[source_key][status_key] == "right_censored" for r in merged)
            recovered = sum(r[source_key][status_key] == "eligible" and r[source_key][rec_key] is True for r in merged)
            not_recovered = sum(r[source_key][status_key] == "eligible" and r[source_key][rec_key] is False for r in merged)
            rec_rows.append({
                "source": source_name,
                "layer": layer_name,
                "deviations": deviations,
                "eligible": eligible,
                "right_censored": censored,
                "recovered": recovered,
                "not_recovered": not_recovered,
                "recovery_rate": recovered / eligible if eligible else np.nan,
            })
    recovery_df = pd.DataFrame(rec_rows)
    recovery_df.to_csv(out / "table4_rq4_recovery_denominators.csv", index=False, float_format="%.4f")

    # ----------------------- Table 5: Direction distribution ---------------
    dir_rows: list[dict[str, Any]] = []
    for source_key, source_name in (("a1", "Annotator 1"), ("a2", "Annotator 2"), ("adj", "Adjudicated"), ("ptta", "PTTA")):
        cnt = Counter(r[source_key]["direction"] for r in merged)
        for d in ("attacker_aligned", "defensive", "incidental", "none"):
            dir_rows.append({"source": source_name, "direction": d, "count": cnt.get(d, 0), "proportion": cnt.get(d, 0) / len(merged)})
    direction_df = pd.DataFrame(dir_rows)
    direction_df.to_csv(out / "table5_rq4_direction_distribution.csv", index=False, float_format="%.4f")

    # ----------------------- Table 6: ASR endpoint baseline ----------------
    asr_pred = [bool(r["ptta"].get("strict_asr")) for r in merged]
    asr_rows: list[dict[str, Any]] = []
    for label, field in (("Structural Deviation", "structural_deviation"), ("Parameter Deviation", "parameter_deviation")):
        yt = [bool(r["adj"][field]) for r in merged]
        sc = binary_scores(yt, asr_pred)
        asr_rows.append({"target": label, "n": len(merged), "accuracy": sc["accuracy"], "kappa": sc["kappa"], "precision": sc["precision"], "recall": sc["recall"], "f1": sc["f1"]})
    pd.DataFrame(asr_rows).to_csv(out / "table6_rq4_asr_baseline.csv", index=False, float_format="%.4f")

    # ----------------------- Tables 7/8: real sensitivity ------------------
    sens_rec_rows: list[dict[str, Any]] = []
    for k, v in sorted(sensitivity.get("recovery_window", {}).items(), key=lambda kv: int(kv[0])):
        sens_rec_rows.append({
            "recovery_window": int(k),
            "eligible": v.get("eligible"),
            "recovered": v.get("recovered"),
            "recovery_rate": v.get("recovery_rate"),
        })
    sens_recovery_df = pd.DataFrame(sens_rec_rows)
    sens_recovery_df.to_csv(out / "table7_rq4_sensitivity_recovery_window.csv", index=False, float_format="%.4f")

    sens_arg_rows: list[dict[str, Any]] = []
    for k, v in sorted(sensitivity.get("alignment_weights", {}).items(), key=lambda kv: float(kv[0])):
        sens_arg_rows.append({
            "arg_change_weight": float(k),
            "structural_deviation_count": v.get("structural_deviation_count"),
            "parameter_deviation_count": v.get("parameter_deviation_count"),
            "note": v.get("note"),
        })
    sens_arg_df = pd.DataFrame(sens_arg_rows)
    sens_arg_df.to_csv(out / "table8_rq4_sensitivity_arg_weight.csv", index=False)

    # ----------------------- Table 9: sampling audit -----------------------
    audit_rows: list[dict[str, Any]] = []
    for variable, expected_map in manifest.get("stratification", {}).items():
        if not isinstance(expected_map, dict):
            continue
        for level, expected in expected_map.items():
            if level == "note" or not isinstance(expected, (int, float)):
                continue
            observed = None
            if variable == "by_source":
                observed = sum(r["source_rq"] == level for r in merged)
            elif variable == "by_success":
                target_bool = str(level).lower() == "true"
                observed = sum(bool(r["ptta"].get("strict_asr")) == target_bool for r in merged)
            elif variable == "by_length_group":
                observed = sum(r["length_group"] == level for r in merged)
            elif variable == "by_template":
                observed = sum(r["template_family"] == level for r in merged)
            elif variable == "by_family":
                observed = sum(r["workflow_family"] == level for r in merged)
            audit_rows.append({"stratum": variable, "level": level, "expected": expected, "observed": observed, "match": (observed == expected) if observed is not None else None})
    pd.DataFrame(audit_rows).to_csv(out / "table9_rq4_sampling_audit.csv", index=False)

    # ----------------------- Disagreement supplements ----------------------
    ptta_dis: list[dict[str, Any]] = []
    for r in merged:
        diff_fields = []
        for f in ("structural_deviation", "parameter_deviation", "first_deviation_position", "direction", "structural_recovery_status", "structural_stable_recovery", "parameter_recovery_status", "parameter_stable_recovery"):
            if r["adj"].get(f) != r["ptta"].get(f):
                diff_fields.append(f)
        if diff_fields:
            ptta_dis.append({
                "pair_id": r["pair_id"],
                "source_rq": r["source_rq"],
                "task_id": r["task_id"],
                "differing_fields": ";".join(diff_fields),
                "adjudicated_direction": r["adj"].get("direction"),
                "ptta_direction": r["ptta"].get("direction"),
                "adjudicated_structural_deviation": r["adj"].get("structural_deviation"),
                "ptta_structural_deviation": r["ptta"].get("structural_deviation"),
                "adjudicated_parameter_deviation": r["adj"].get("parameter_deviation"),
                "ptta_parameter_deviation": r["ptta"].get("parameter_deviation"),
            })
    pd.DataFrame(ptta_dis).to_csv(out / "supplementary_ptta_disagreements.csv", index=False)

    ann_dis: list[dict[str, Any]] = []
    for r in merged:
        diff_fields = []
        for f in ("structural_deviation", "parameter_deviation", "first_deviation_position", "direction", "structural_recovery_status", "structural_stable_recovery", "parameter_recovery_status", "parameter_stable_recovery"):
            if r["a1"].get(f) != r["a2"].get(f):
                diff_fields.append(f)
        if diff_fields:
            ann_dis.append({"pair_id": r["pair_id"], "source_rq": r["source_rq"], "task_id": r["task_id"], "differing_fields": ";".join(diff_fields), "a1_confidence": r["a1"].get("confidence"), "a2_confidence": r["a2"].get("confidence")})
    pd.DataFrame(ann_dis).to_csv(out / "supplementary_annotator_disagreements.csv", index=False)

    # --------------------- Task-cluster bootstrap CIs ----------------------
    bootstrap_defs: list[tuple[str, Callable[[Sequence[dict[str, Any]]], float]]] = [
        ("Structural_Deviation_F1", lambda rr: binary_scores([bool(x["adj"]["structural_deviation"]) for x in rr], [bool(x["ptta"]["structural_deviation"]) for x in rr])["f1"]),
        ("Parameter_Deviation_F1", lambda rr: binary_scores([bool(x["adj"]["parameter_deviation"]) for x in rr], [bool(x["ptta"]["parameter_deviation"]) for x in rr])["f1"]),
        ("Direction_Kappa", lambda rr: cohens_kappa([x["adj"]["direction"] for x in rr], [x["ptta"]["direction"] for x in rr])),
        ("Direction_MacroF1", lambda rr: macro_f1([x["adj"]["direction"] for x in rr], [x["ptta"]["direction"] for x in rr])),
    ]

    def fdp_mae_metric(rr: Sequence[dict[str, Any]]) -> float:
        sub = [x for x in rr if x["adj"].get("first_deviation_position") is not None and x["ptta"].get("first_deviation_position") is not None]
        if not sub:
            return float("nan")
        return float(np.mean([abs(int(x["adj"]["first_deviation_position"]) - int(x["ptta"]["first_deviation_position"])) for x in sub]))

    bootstrap_defs.append(("FDP_MAE", fdp_mae_metric))

    for layer, label in (("structural", "Structural"), ("parameter", "Parameter")):
        def make_recovery_f1(layer_name: str) -> Callable[[Sequence[dict[str, Any]]], float]:
            def fn(rr: Sequence[dict[str, Any]]) -> float:
                sub = rows_joint_eligible(rr, layer_name, "adj", "ptta")
                if not sub:
                    return float("nan")
                field = f"{layer_name}_stable_recovery"
                return binary_scores([bool(x["adj"][field]) for x in sub], [bool(x["ptta"][field]) for x in sub])["f1"]
            return fn
        bootstrap_defs.append((f"{label}_Recovery_F1_joint_eligible", make_recovery_f1(layer)))

    boot_rows = []
    for i, (name, fn) in enumerate(bootstrap_defs):
        br = cluster_bootstrap(merged, fn, cluster_key="cluster_id", B=args.bootstrap, seed=args.seed + 1000 * i)
        boot_rows.append({
            "metric": name,
            "estimate": round_or_none(br.estimate, 5),
            "ci_low": round_or_none(br.ci_low, 5),
            "ci_high": round_or_none(br.ci_high, 5),
            "valid_resamples": br.valid_resamples,
            "bootstrap_unit": "source_rq::task_id",
        })
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out / "bootstrap_core_metrics.csv", index=False)

    # ----------------------------- figures ---------------------------------
    make_figures(out, iaa_df, ptta_df, yt_dir, yp_dir, recovery_df, sens_recovery_df)

    # ----------------------------- summary ---------------------------------
    def row_metric(df: pd.DataFrame, metric: str, col: str) -> float | None:
        hit = df[df["metric"] == metric]
        if hit.empty:
            return None
        return finite_or_none(hit.iloc[0][col])

    summary = {
        "n_pairs": len(merged),
        "n_task_clusters": len({r["cluster_id"] for r in merged}),
        "strict_asr_successes": strict_asr_observed,
        "validation": validation_report["status"],
        "core_results": {
            "structural_deviation_f1": row_metric(ptta_df, "Structural Deviation", "f1"),
            "parameter_deviation_f1": row_metric(ptta_df, "Parameter Deviation", "f1"),
            "direction_kappa": row_metric(ptta_df, "Direction", "kappa"),
            "direction_macro_f1": row_metric(ptta_df, "Direction", "macro_f1"),
            "fdp_mae": finite_or_none(fdp_row["mae"]),
            "fdp_exact": finite_or_none(fdp_row["exact_match_rate"]),
            "structural_recovery_f1_joint_eligible": row_metric(ptta_df, "Structural Stable Recovery (joint eligible)", "f1"),
            "parameter_recovery_f1_joint_eligible": row_metric(ptta_df, "Parameter Stable Recovery (joint eligible)", "f1"),
        },
        "interpretation_notes": [
            "Adjudicated human labels are the reference; A1/A2 are used only for inter-annotator agreement.",
            "Stable-recovery F1 is conditional on joint eligibility; status agreement is reported separately.",
            "Right-censored/N/A cases are excluded from stable-recovery denominators.",
            "Sensitivity values are read from the supplied JSON and are not simulated.",
            *sensitivity_notes,
        ],
    }
    write_json(out / "rq4_summary.json", summary)

    print("RQ4 formal analysis completed.")
    print(f"Output directory: {out.resolve()}")
    print(json.dumps(summary["core_results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
