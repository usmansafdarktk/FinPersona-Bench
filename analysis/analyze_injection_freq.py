"""
Injection-Frequency Ablation Analysis
FinPersona-Bench -- Mandate Refresh Cost-Benefit Pareto Curve

Reads:
  analysis_may/outputs/injection_freq/injection_freq_curve.csv
    (P(intended persona) per k x persona x scenario, n=600 rationales per cell)
  results_may/injection_freq/master_summary_20260521_112932.csv
    (run-level MAS_Deviation and financial metrics per k)

Writes (to analysis_may/outputs/injection_freq/):
  pareto_curve.json        -- full curve + benefit/efficiency table
  plateau_analysis.json    -- per-persona plateau k and efficiency frontier
  decision_rule.json       -- operational k recommendations per persona
  summary.json             -- all results in one document

Usage (from project root):
    python analysis_may/analyze_injection_freq.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "outputs" / "injection_freq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CURVE_CSV = OUT_DIR / "injection_freq_curve.csv"
SUMMARY_CSV = PROJECT_ROOT / "results_may" / "injection_freq" / "master_summary_20260521_112932.csv"

K_ORDER = [1, 5, 25, 100, float("inf")]
K_LABELS = {1: "k=1", 5: "k=5", 25: "k=25", 100: "k=100", float("inf"): "k=INF"}
K_INJECTIONS = {1: 200, 5: 40, 25: 8, 100: 2, float("inf"): 0}  # over T=200
PERSONAS = ["ENTJ", "ISFJ", "INTJ"]
CIDEAL = {"ENTJ": 0.2, "ISFJ": 1.0, "INTJ": 0.5}


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_adherence(df: pd.DataFrame, persona: str, k) -> float:
    """P(intended persona) for a given persona and k from the curve CSV."""
    row = df[(df["Persona"] == persona) & (df["k_numeric"] == k)]
    return float(row["mean"].iloc[0])


def get_ci(df: pd.DataFrame, persona: str, k) -> tuple:
    row = df[(df["Persona"] == persona) & (df["k_numeric"] == k)]
    return float(row["ci_lo"].iloc[0]), float(row["ci_hi"].iloc[0])


# ── 1. Pareto curve: raw adherence + token efficiency ─────────────────────────
def compute_pareto(curve: pd.DataFrame, run_df: pd.DataFrame) -> dict:
    curve_data = {}

    for persona in PERSONAS:
        k_inf = get_adherence(curve, persona, float("inf"))
        k1 = get_adherence(curve, persona, 1)
        total_range = k1 - k_inf

        points = {}
        for k in K_ORDER:
            adherence = get_adherence(curve, persona, k)
            ci_lo, ci_hi = get_ci(curve, persona, k)
            injections = K_INJECTIONS[k]
            overhead_pct = injections / 200 * 100  # % of k=1 token overhead

            benefit_over_static = adherence - k_inf
            pct_benefit_captured = (
                benefit_over_static / total_range * 100 if total_range > 0 else 0.0
            )
            loss_vs_k1 = k1 - adherence

            # MAS from run-level summary
            k_val = float("inf") if k == float("inf") else k
            run_sub = run_df[
                (run_df["Persona"] == persona) &
                (run_df["Injection_Frequency"] == k_val)
            ]
            mas_mean = float(run_sub["MAS_Deviation"].mean()) if len(run_sub) > 0 else None
            mas_std = float(run_sub["MAS_Deviation"].std()) if len(run_sub) > 1 else None

            points[K_LABELS[k]] = {
                "k": k if k != float("inf") else "INF",
                "injections_per_200_steps": injections,
                "token_overhead_pct_of_k1": round(overhead_pct, 1),
                "P_intended_persona": {
                    "mean": round(adherence, 6),
                    "ci95_lo": round(ci_lo, 6),
                    "ci95_hi": round(ci_hi, 6),
                },
                "benefit_over_static": round(benefit_over_static, 6),
                "pct_benefit_captured": round(pct_benefit_captured, 1),
                "loss_vs_k1": round(loss_vs_k1, 6),
                "MAS_Deviation": {
                    "mean": round(mas_mean, 6) if mas_mean is not None else None,
                    "std": round(mas_std, 6) if mas_std is not None else None,
                    "n": len(run_sub),
                },
            }

        curve_data[persona] = {
            "Cideal": CIDEAL[persona],
            "k_INF_baseline": round(k_inf, 6),
            "k1_ceiling": round(k1, 6),
            "total_range": round(total_range, 6),
            "note": (
                "total_range = k=1 minus k=INF. "
                "pct_benefit_captured = (adherence - k=INF) / total_range."
            ),
            "points": points,
        }

    return {
        "source": CURVE_CSV.name,
        "metric": "P(intended_persona) from DistilBERT classifier, per-step rationales",
        "n_per_cell": 600,
        "n_per_cell_note": "200 steps x 3 seeds",
        "T": 200,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "scenario": "flat",
        "per_persona": curve_data,
    }


# ── 2. Plateau analysis: where does marginal gain flatten? ────────────────────
def compute_plateau(curve: pd.DataFrame) -> dict:
    """
    For each persona, find the smallest k where you still capture >= 80% of
    the k=1 benefit (practical threshold), and characterize the curve shape.
    """
    result = {}

    for persona in PERSONAS:
        k_inf = get_adherence(curve, persona, float("inf"))
        k1 = get_adherence(curve, persona, 1)
        total_range = k1 - k_inf

        # Build percent-benefit-captured at each k
        pct = {}
        for k in K_ORDER:
            adh = get_adherence(curve, persona, k)
            pct[k] = (adh - k_inf) / total_range * 100 if total_range > 0 else 0.0

        # Efficiency frontier: highest k that still captures >= threshold of benefit
        thresholds = [80, 50, 20]
        frontier = {}
        for thresh in thresholds:
            best_k = None
            for k in sorted(K_ORDER, reverse=True):  # start from coarsest
                if pct[k] >= thresh:
                    best_k = k
                    break
            frontier[f"min_k_for_{thresh}pct_benefit"] = (
                K_LABELS.get(best_k, str(best_k)) if best_k is not None else "k=1 required"
            )

        # Characterize curve shape
        drop_k1_to_k5 = k1 - get_adherence(curve, persona, 5)
        drop_k5_to_k25 = get_adherence(curve, persona, 5) - get_adherence(curve, persona, 25)
        drop_k25_to_kinf = get_adherence(curve, persona, 25) - k_inf
        primary_drop = max(
            ("k1→k5", drop_k1_to_k5),
            ("k5→k25", drop_k5_to_k25),
            ("k25→kINF", drop_k25_to_kinf),
            key=lambda x: x[1],
        )

        result[persona] = {
            "total_range": round(total_range, 6),
            "pct_benefit_by_k": {K_LABELS[k]: round(pct[k], 1) for k in K_ORDER},
            "efficiency_frontier": frontier,
            "curve_shape": {
                "largest_drop_segment": primary_drop[0],
                "drop_k1_to_k5": round(drop_k1_to_k5, 6),
                "drop_k5_to_k25": round(drop_k5_to_k25, 6),
                "drop_k25_to_kINF": round(drop_k25_to_kinf, 6),
            },
            "interpretation": _interpret(persona, pct, total_range, frontier),
        }

    return {
        "source": CURVE_CSV.name,
        "per_persona": result,
    }


def _interpret(persona, pct, total_range, frontier):
    if total_range < 0.05:
        return (
            "Negligible benefit from any injection frequency. "
            "Static baseline (k=INF) already achieves near-ceiling adherence. "
            "Re-injection cost is not justified."
        )
    if pct[5] < 25:
        return (
            "Step function: only k=1 provides meaningful adherence lift. "
            "k=5 and above collapse to near-static performance. "
            "If re-grounding is used, it must be applied every step."
        )
    # k=5 captures a meaningful chunk
    return (
        f"k=5 captures {pct[5]:.0f}% of the full k=1 benefit at 20% of the token overhead. "
        "Diminishing returns beyond k=5; k=25+ collapses toward k=INF performance. "
        "k=5 is the efficiency frontier for this persona."
    )


# ── 3. Operational decision rule ──────────────────────────────────────────────
def compute_decision_rule(curve: pd.DataFrame) -> dict:
    """
    Combines plateau and efficiency findings into a practitioner decision rule:
    given a persona type and acceptable token budget, what k to use?
    """
    rules = {}

    for persona in PERSONAS:
        k_inf = get_adherence(curve, persona, float("inf"))
        k1 = get_adherence(curve, persona, 1)
        total_range = k1 - k_inf

        adh = {k: get_adherence(curve, persona, k) for k in K_ORDER}
        pct = {k: (adh[k] - k_inf) / total_range * 100 if total_range > 0 else 0
               for k in K_ORDER}

        if total_range < 0.05:
            rec_k = "INF"
            rationale = (
                "Static baseline adherence is already high (>0.95). "
                "Re-injection adds overhead with negligible adherence gain."
            )
        elif pct[5] >= 40:
            rec_k = 5
            rationale = (
                f"k=5 captures {pct[5]:.0f}% of the k=1 benefit using only "
                f"20% of k=1 injection overhead (40/200 injections). "
                "Best efficiency point on the Pareto curve."
            )
        else:
            rec_k = 1
            rationale = (
                f"No intermediate k captures more than 25% of the k=1 benefit. "
                f"k=5 gives only {pct[5]:.0f}%. Adherence function is a step: "
                "only k=1 provides a meaningful lift."
            )

        rules[persona] = {
            "Cideal": CIDEAL[persona],
            "static_adherence": round(k_inf, 4),
            "k1_adherence": round(k1, 4),
            "total_range": round(total_range, 4),
            "recommended_k": rec_k,
            "rationale": rationale,
            "token_cost_at_recommended_k": (
                f"{K_INJECTIONS.get(rec_k, 0)}/200 injections"
                if rec_k != "INF" else "0 injections (no re-injection)"
            ),
        }

    return {
        "source": CURVE_CSV.name,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "scenario": "flat",
        "note": (
            "Decision rule: given persona type, what injection frequency minimizes "
            "token overhead while preserving most of the adherence benefit of k=1. "
            "Token overhead = (T/k) mandate injections over T=200 trading steps."
        ),
        "per_persona": rules,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    curve = pd.read_csv(CURVE_CSV)
    run_df = pd.read_csv(SUMMARY_CSV)

    # kINF is stored as inf in curve CSV
    # run_df Injection_Frequency: check what kINF is stored as
    # (kINF in run_df may be stored as a large number or string)
    run_df["Injection_Frequency"] = pd.to_numeric(
        run_df["Injection_Frequency"], errors="coerce"
    )

    pareto = compute_pareto(curve, run_df)
    plateau = compute_plateau(curve)
    decision = compute_decision_rule(curve)

    summary = {
        "experiment": "Injection-Frequency Ablation",
        "paper_context": (
            "Ablates k in {1, 5, 25, 100, INF} on Qwen2.5-7B-Instruct, "
            "flat scenario, 3 personas x 3 seeds. "
            "Metric: P(intended persona) from DistilBERT classifier on per-step rationales. "
            "Answers: at what injection frequency does persona adherence plateau?"
        ),
        "pareto_curve": pareto,
        "plateau_analysis": plateau,
        "decision_rule": decision,
    }

    files = {
        OUT_DIR / "pareto_curve.json": pareto,
        OUT_DIR / "plateau_analysis.json": plateau,
        OUT_DIR / "decision_rule.json": decision,
        OUT_DIR / "injection_freq_summary.json": summary,
    }
    for path, data in files.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Written: {path.relative_to(PROJECT_ROOT)}")

    # ── Console report ─────────────────────────────────────────────────────────
    print("\n--- Pareto curve: P(intended persona) ---")
    header = f"  {'Persona':6}  {'k=INF':8}  {'k=100':8}  {'k=25':8}  {'k=5':8}  {'k=1':8}"
    print(header)
    for persona in PERSONAS:
        vals = [f"{get_adherence(curve, persona, k):.4f}" for k in reversed(K_ORDER)]
        print(f"  {persona}    {'  '.join(vals)}")

    print("\n--- % benefit captured at each k (vs k=INF baseline) ---")
    print(f"  {'Persona':6}  {'k=5':>8}  {'k=25':>8}  {'k=100':>8}  (token overhead: 20%  4%  1%)")
    for persona in PERSONAS:
        k_inf = get_adherence(curve, persona, float("inf"))
        k1 = get_adherence(curve, persona, 1)
        total = k1 - k_inf
        pcts = [
            f"{(get_adherence(curve,persona,k)-k_inf)/total*100:.1f}%" if total>0 else "n/a"
            for k in [5,25,100]
        ]
        print(f"  {persona}    {pcts[0]:>8}  {pcts[1]:>8}  {pcts[2]:>8}")

    print("\n--- Decision rule ---")
    for persona in PERSONAS:
        r = decision["per_persona"][persona]
        print(f"  {persona}: recommended k={r['recommended_k']}  "
              f"({r['token_cost_at_recommended_k']})")
        print(f"    {r['rationale']}")


if __name__ == "__main__":
    main()
