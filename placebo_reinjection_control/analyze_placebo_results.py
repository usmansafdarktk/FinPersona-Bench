"""
Placebo RCT Results Analysis
FinPersona-Bench -- Causal Identification of Mandate Salience

Reads:
  placebo_results/placebo_summary_*.csv                (3-arm MAS data: static, placebo, memory)
  placebo_results/persona_classifier/metrics.json      (classifier accuracies)
  results_april/initial_general_results/               (April raw CSVs for baseline comparison)

Writes:
  placebo_results/analysis/mas_deviation.json          (mean/std per persona x arm, placebo experiment)
  placebo_results/analysis/april_baseline.json         (April static+memory for claude-sonnet-4-6, flat)
  placebo_results/analysis/three_way_comparison.json   (April static, April memory, Placebo -- aligned)
  placebo_results/analysis/wilcoxon_tests.json         (3 paired tests)
  placebo_results/analysis/classifier_sanity.json      (P(intended) per arm)
  placebo_results/analysis/summary.json                (all results in one place)

Usage (from project root):
    python placebo_reinjection_control/analyze_placebo_results.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
RESULTS_DIR = HERE / "placebo_results"
OUT_DIR = RESULTS_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick the most recent summary (deterministic: sort and take last)
summary_files = sorted(RESULTS_DIR.glob("placebo_summary_*.csv"))
if not summary_files:
    raise FileNotFoundError(f"No placebo_summary_*.csv found in {RESULTS_DIR}")
SUMMARY_CSV = summary_files[-1]

CLASSIFIER_JSON = RESULTS_DIR / "persona_classifier" / "metrics.json"

# April raw CSVs: static and memory arms for claude-sonnet-4-6, flat scenario
APRIL_BASE = HERE.parent / "results_april" / "initial_general_results" / "claude-sonnet-4-6" / "flat"

ARM_ORDER = ["static", "placebo", "memory"]
PERSONA_ORDER = ["ENTJ", "ISFJ", "INTJ"]
CIDEAL = {"ENTJ": 0.2, "ISFJ": 1.0, "INTJ": 0.5}
SEEDS = [42, 123, 456, 789, 999]


def load_data():
    df = pd.read_csv(SUMMARY_CSV)
    with open(CLASSIFIER_JSON) as f:
        clf = json.load(f)
    return df, clf


# ── April baseline: compute MAS from raw per-step CSVs ────────────────────────
def compute_april_baseline() -> dict:
    """
    Reads the 30 per-step CSVs for claude-sonnet-4-6 / flat from results_april
    and computes MAS Deviation = mean(|cash_frac - Cideal|) per run.
    """
    rows = []
    missing = []
    for arm in ["static", "memory"]:
        for persona in PERSONA_ORDER:
            for seed in SEEDS:
                path = APRIL_BASE / f"seed{seed}" / f"{persona}_{arm}_flat_seed{seed}.csv"
                if not path.exists():
                    missing.append(str(path))
                    continue
                df = pd.read_csv(path)
                cash_frac = df["Cash"] / df["Portfolio_Value"]
                mas = float((cash_frac - CIDEAL[persona]).abs().mean())
                rows.append({"Persona": persona, "Arm": arm, "Seed": seed, "MAS_Deviation": mas})

    df_runs = pd.DataFrame(rows)

    per_persona = {}
    for persona in PERSONA_ORDER:
        per_persona[persona] = {"Cideal": CIDEAL[persona], "arms": {}}
        for arm in ["static", "memory"]:
            vals = df_runs[(df_runs.Persona == persona) & (df_runs.Arm == arm)]["MAS_Deviation"]
            per_persona[persona]["arms"][arm] = {
                "mean": round(float(vals.mean()), 6),
                "std": round(float(vals.std()), 6),
                "n": int(len(vals)),
                "per_seed": {
                    str(seed): round(float(v), 6)
                    for seed, v in zip(
                        vals.index.map(
                            lambda i: df_runs.loc[i, "Seed"]
                        ),
                        vals,
                    )
                },
            }

    # Simpler per-seed mapping
    for persona in PERSONA_ORDER:
        for arm in ["static", "memory"]:
            sub = df_runs[(df_runs.Persona == persona) & (df_runs.Arm == arm)]
            per_persona[persona]["arms"][arm]["per_seed"] = {
                str(int(r.Seed)): round(r.MAS_Deviation, 6)
                for _, r in sub.iterrows()
            }

    return {
        "source": "results_april/initial_general_results/claude-sonnet-4-6/flat/",
        "model": "claude-sonnet-4-6",
        "scenario": "flat",
        "metric": "MAS_Deviation",
        "note": "Computed from raw per-step CSVs: mean(|Cash/Portfolio_Value - Cideal|) per run.",
        "missing_files": missing,
        "per_persona": per_persona,
    }


# ── Three-way comparison table ─────────────────────────────────────────────────
def compute_three_way(april: dict, placebo_mas: dict) -> dict:
    """
    Aligns April static/memory with placebo arm into one comparison table.
    All three use the same model (claude-sonnet-4-6), scenario (flat), seeds.
    """
    table = {}
    for persona in PERSONA_ORDER:
        april_arms = april["per_persona"][persona]["arms"]
        placebo_arms = placebo_mas["per_persona"][persona]["arms"]
        table[persona] = {
            "Cideal": CIDEAL[persona],
            "april_static": april_arms.get("static", {}),
            "placebo": placebo_arms.get("placebo", {}),
            "april_memory": april_arms.get("memory", {}),
        }

    return {
        "note": (
            "All three arms use claude-sonnet-4-6, flat scenario, seeds [42,123,456,789,999]. "
            "april_static and april_memory come from results_april raw CSVs; "
            "placebo comes from placebo_results summary CSV."
        ),
        "arms": {
            "april_static": "Mandate at init only (April experiment)",
            "placebo": "Boilerplate re-injected each step (Placebo experiment)",
            "april_memory": "Mandate re-injected each step (April experiment)",
        },
        "per_persona": table,
    }


# ── 1. MAS Deviation per persona x arm ────────────────────────────────────────
def compute_mas_deviation(df: pd.DataFrame) -> dict:
    agg = (
        df.groupby(["Persona", "Arm"])["MAS_Deviation"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    per_persona = {}
    for persona in PERSONA_ORDER:
        cideal = df.loc[df["Persona"] == persona, "Cideal"].iloc[0]
        per_persona[persona] = {"Cideal": float(cideal), "arms": {}}
        for arm in ARM_ORDER:
            row = agg[(agg["Persona"] == persona) & (agg["Arm"] == arm)]
            if row.empty:
                continue
            per_persona[persona]["arms"][arm] = {
                "mean": round(float(row["mean"].iloc[0]), 6),
                "std": round(float(row["std"].iloc[0]), 6),
                "n": int(row["n"].iloc[0]),
            }

    # Grand mean across all personas (collapsed)
    grand = {}
    for arm in ARM_ORDER:
        vals = df.loc[df["Arm"] == arm, "MAS_Deviation"]
        grand[arm] = {
            "mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std()), 6),
            "n": int(len(vals)),
        }

    return {
        "source_file": SUMMARY_CSV.name,
        "metric": "MAS_Deviation",
        "note": (
            "MAS Deviation = |actual_cash_fraction - Cideal|. "
            "Lower = better adherence to persona mandate."
        ),
        "per_persona": per_persona,
        "grand_mean": grand,
    }


# ── 2. Paired Wilcoxon signed-rank tests ──────────────────────────────────────
def compute_wilcoxon(df: pd.DataFrame) -> dict:
    pivot = df.pivot_table(
        index=["Persona", "Seed"], columns="Arm", values="MAS_Deviation"
    )

    comparisons = [
        ("static", "placebo", "Static vs. Placebo"),
        ("placebo", "memory", "Placebo vs. Memory"),
        ("static", "memory", "Static vs. Memory"),
    ]

    results = {}
    for a, b, label in comparisons:
        diffs = pivot[a] - pivot[b]
        stat, p = wilcoxon(diffs, alternative="two-sided")
        results[label] = {
            "arm_A": a,
            "arm_B": b,
            "test": "Wilcoxon signed-rank (paired, two-tailed)",
            "n_pairs": int(len(diffs)),
            "pairing": "Persona x Seed",
            "W": float(stat),
            "p_value": round(float(p), 6),
            "significant_at_0.05": bool(p < 0.05),
            "mean_diff_A_minus_B": round(float(diffs.mean()), 6),
            "interpretation": (
                f"{a} MAS {'higher' if diffs.mean() > 0 else 'lower'} than {b} "
                f"by {abs(diffs.mean()):.4f} on average"
            ),
        }

    return {
        "source_file": SUMMARY_CSV.name,
        "metric": "MAS_Deviation",
        "note": (
            "Positive mean_diff means arm_A has higher (worse) MAS deviation. "
            "n=15 pairs (3 personas x 5 seeds)."
        ),
        "tests": results,
    }


# ── 3. Classifier sanity check ─────────────────────────────────────────────────
def compute_classifier_sanity(clf: dict) -> dict:
    arm_map = {
        "static": clf.get("static_accuracy"),
        "placebo": clf.get("placebo_accuracy"),
        "memory": clf.get("memory_accuracy"),
    }

    return {
        "source_file": "persona_classifier/metrics.json",
        "model": clf.get("model"),
        "personas": clf.get("personas"),
        "metric": "P(intended_persona) -- classifier accuracy on rationales",
        "note": (
            "Sanity check: placebo should sit near static (~66%), "
            "not near memory (~87%), confirming boilerplate did not "
            "accidentally activate persona-associated language."
        ),
        "per_arm": {
            arm: {"accuracy": round(float(v), 6) if v is not None else None}
            for arm, v in arm_map.items()
        },
        "sanity_check_passed": bool(
            arm_map["placebo"] is not None
            and arm_map["static"] is not None
            and arm_map["memory"] is not None
            and abs(arm_map["placebo"] - arm_map["static"])
            < abs(arm_map["placebo"] - arm_map["memory"])
        ),
    }


# ── 4. Write outputs ───────────────────────────────────────────────────────────
def main():
    df, clf = load_data()

    mas = compute_mas_deviation(df)
    wilc = compute_wilcoxon(df)
    sanity = compute_classifier_sanity(clf)
    april = compute_april_baseline()
    three_way = compute_three_way(april, mas)

    summary = {
        "experiment": "Placebo Re-injection Control (3-Arm RCT)",
        "paper_context": (
            "Addresses Reviewer VG7b W1: content vs. recency confound. "
            "Arms: static (mandate at init), placebo (boilerplate re-injected), "
            "memory (mandate re-injected). Model: claude-sonnet-4-6, scenario: flat, "
            "3 personas x 5 seeds, T=200."
        ),
        "mas_deviation": mas,
        "april_baseline": april,
        "three_way_comparison": three_way,
        "wilcoxon_tests": wilc,
        "classifier_sanity": sanity,
    }

    files = {
        OUT_DIR / "mas_deviation.json": mas,
        OUT_DIR / "april_baseline.json": april,
        OUT_DIR / "three_way_comparison.json": three_way,
        OUT_DIR / "wilcoxon_tests.json": wilc,
        OUT_DIR / "classifier_sanity.json": sanity,
        OUT_DIR / "summary.json": summary,
    }

    for path, data in files.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Written: {path.relative_to(HERE.parent)}")

    # ── Quick console report ───────────────────────────────────────────────────
    print("\n--- Placebo experiment MAS Deviation (mean +/- std) ---")
    for persona in PERSONA_ORDER:
        arms = mas["per_persona"][persona]["arms"]
        row = "  ".join(
            f"{arm}: {arms[arm]['mean']:.4f}+/-{arms[arm]['std']:.4f}"
            for arm in ARM_ORDER
            if arm in arms
        )
        print(f"  {persona}: {row}")

    print("\n--- Three-way comparison: April static / Placebo / April memory ---")
    header = f"  {'Persona':6}  {'april_static':20}  {'placebo':20}  {'april_memory':20}"
    print(header)
    for persona in PERSONA_ORDER:
        p = three_way["per_persona"][persona]
        def fmt(d): return f"{d['mean']:.4f}+/-{d['std']:.4f}" if d else "n/a"
        print(f"  {persona:6}  {fmt(p['april_static']):20}  {fmt(p['placebo']):20}  {fmt(p['april_memory']):20}")

    print("\n--- Wilcoxon tests (placebo experiment arms) ---")
    for label, res in wilc["tests"].items():
        sig = "*" if res["significant_at_0.05"] else "ns"
        print(f"  {label}: W={res['W']:.1f}, p={res['p_value']:.4f} ({sig})")

    print("\n--- Classifier sanity ---")
    for arm, v in sanity["per_arm"].items():
        print(f"  {arm}: {v['accuracy']*100:.1f}%")
    passed = "PASSED" if sanity["sanity_check_passed"] else "FAILED"
    print(f"  Sanity check: {passed}")


if __name__ == "__main__":
    main()
