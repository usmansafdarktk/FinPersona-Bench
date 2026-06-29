"""
OCEAN Experiment Results Analysis
FinPersona-Bench -- Framework-Independence Replication

Reads:
  results_ocean/master_summary_20260524_194647.csv
  results_april/initial_general_results/  (MBTI baseline for three-way comparison)

Writes (to analysis_ocean/outputs/):
  ocean_flat_mas.json          -- MAS/cash/return per persona x model x arm, flat scenario
  ocean_all_scenarios_mas.json -- Same but collapsed across scenarios (O1/O2 only)
  bidirectionality_check.json  -- Does memory help O1 and hurt O2? Per model.
  o3_ablation.json             -- O3 numerical-only flat results
  three_way_gap_table.json     -- MBTI vs OCEAN vs O3 gap table (flat, aligned)
  summary.json                 -- All of the above in one document

Usage (from project root):
    python analysis_ocean/analyze_ocean_results.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OCEAN_CSV = sorted(
    (PROJECT_ROOT / "results_ocean").glob("master_summary_*.csv")
)[-1]

APRIL_BASE = (
    PROJECT_ROOT / "results_april" / "initial_general_results" / "claude-sonnet-4-6" / "flat"
)

MODELS = ["claude-sonnet-4-6", "gpt-4o-mini", "gemini-2.5-flash"]
SEEDS = [42, 123, 456, 789, 999]
CIDEAL = {"O1_conservative": 1.0, "O2_aggressive": 0.2,
          "O3_conservative": 1.0, "O3_aggressive": 0.2}
MBTI_CIDEAL = {"ISFJ": 1.0, "ENTJ": 0.2}

METRICS = ["MAS_Deviation", "Avg_Cash_Pct", "Return_Pct", "Rationality_Score"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def agg_stats(series: pd.Series) -> dict:
    """Mean, std, and 95% bootstrap CI for a series of values."""
    vals = series.dropna().values
    if len(vals) == 0:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None, "n": 0}
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if len(vals) >= 3:
        res = bootstrap(
            (vals,), np.mean, n_resamples=5000, confidence_level=0.95,
            random_state=0, method="percentile"
        )
        ci_low = float(res.confidence_interval.low)
        ci_high = float(res.confidence_interval.high)
    else:
        ci_low = ci_high = mean
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95_low": round(ci_low, 6),
        "ci95_high": round(ci_high, 6),
        "n": int(len(vals)),
    }


def per_seed_dict(df_sub: pd.DataFrame, metric: str) -> dict:
    return {
        str(int(r["Seed"])): round(float(r[metric]), 6)
        for _, r in df_sub.iterrows()
    }


# ── 1. OCEAN flat scenario: full per-model per-persona results ─────────────────
def compute_ocean_flat(df: pd.DataFrame) -> dict:
    flat = df[df["Scenario"] == "flat"]
    result = {}

    for persona in ["O1_conservative", "O2_aggressive"]:
        result[persona] = {"Cideal": CIDEAL[persona], "models": {}}
        for model in MODELS:
            result[persona]["models"][model] = {}
            for arm in ["static", "memory"]:
                sub = flat[
                    (flat["Persona"] == persona) &
                    (flat["Model"] == model) &
                    (flat["Agent_Type"] == arm)
                ].sort_values("Seed")
                arms_dict = {}
                for metric in METRICS:
                    arms_dict[metric] = {
                        **agg_stats(sub[metric]),
                        "per_seed": per_seed_dict(sub, metric),
                    }
                result[persona]["models"][model][arm] = arms_dict

    return {
        "source": OCEAN_CSV.name,
        "scenario": "flat",
        "note": (
            "O1_conservative (Cideal=1.0) mirrors ISFJ; "
            "O2_aggressive (Cideal=0.2) mirrors ENTJ. "
            "MAS_Deviation: lower = better adherence. "
            "Avg_Cash_Pct: % cash held. Return_Pct: total return. "
            "Rationality_Score: % value-aligned decisions."
        ),
        "per_persona": result,
    }


# ── 2. OCEAN all scenarios: collapsed across seeds, O1/O2 only ────────────────
def compute_ocean_all_scenarios(df: pd.DataFrame) -> dict:
    result = {}
    for persona in ["O1_conservative", "O2_aggressive"]:
        result[persona] = {"Cideal": CIDEAL[persona], "scenarios": {}}
        for scenario in ["flat", "bull_trap", "crash"]:
            result[persona]["scenarios"][scenario] = {}
            sub_scen = df[(df["Persona"] == persona) & (df["Scenario"] == scenario)]
            for arm in ["static", "memory"]:
                sub = sub_scen[sub_scen["Agent_Type"] == arm]
                result[persona]["scenarios"][scenario][arm] = {
                    metric: agg_stats(sub[metric]) for metric in METRICS
                }
    return {
        "source": OCEAN_CSV.name,
        "note": "Collapsed across all 3 models x 5 seeds = n=15 per cell.",
        "per_persona": result,
    }


# ── 3. Bidirectionality check: per model, flat ────────────────────────────────
def compute_bidirectionality(df: pd.DataFrame) -> dict:
    """
    Primary claim: memory hurts O2_aggressive (MAS goes up) and helps
    O1_conservative (MAS goes down), replicating the MBTI ENTJ/ISFJ pattern.
    Checks this per model and across all models.
    """
    flat = df[df["Scenario"] == "flat"]
    result = {"per_model": {}, "pooled": {}}

    for model in MODELS + ["ALL"]:
        sub = flat if model == "ALL" else flat[flat["Model"] == model]
        n_label = "n=15 (3 models x 5 seeds)" if model == "ALL" else "n=5 seeds"

        entry = {}
        for persona in ["O1_conservative", "O2_aggressive"]:
            static_vals = sub[(sub["Persona"] == persona) & (sub["Agent_Type"] == "static")]["MAS_Deviation"]
            memory_vals = sub[(sub["Persona"] == persona) & (sub["Agent_Type"] == "memory")]["MAS_Deviation"]
            s_mean = float(static_vals.mean())
            m_mean = float(memory_vals.mean())
            delta = round(m_mean - s_mean, 6)
            entry[persona] = {
                "static_mean_MAS": round(s_mean, 6),
                "memory_mean_MAS": round(m_mean, 6),
                "delta_memory_minus_static": delta,
                "direction": "memory HURTS (MAS up)" if delta > 0 else "memory HELPS (MAS down)",
                "n": n_label,
            }

        # Cross-check bidirectionality: O2 hurt AND O1 helped?
        o2_hurt = entry["O2_aggressive"]["delta_memory_minus_static"] > 0
        o1_helped = entry["O1_conservative"]["delta_memory_minus_static"] < 0
        entry["bidirectional"] = o2_hurt and o1_helped
        entry["bidirectional_note"] = (
            "REPLICATES MBTI pattern (O2 hurt + O1 helped)"
            if (o2_hurt and o1_helped)
            else "Does NOT fully replicate"
        )

        if model == "ALL":
            result["pooled"] = entry
        else:
            result["per_model"][model] = entry

    return {
        "source": OCEAN_CSV.name,
        "scenario": "flat",
        "primary_claim": (
            "MBTI replication: memory re-injection hurts aggressive (O2, Cideal=0.2) "
            "and helps conservative (O1, Cideal=1.0). "
            "delta = memory_MAS - static_MAS; positive = memory hurts."
        ),
        "results": result,
    }


# ── 4. O3 ablation: numerical-only personas, flat ─────────────────────────────
def compute_o3_ablation(df: pd.DataFrame) -> dict:
    flat = df[df["Scenario"] == "flat"]
    result = {}

    for persona in ["O3_conservative", "O3_aggressive"]:
        result[persona] = {"Cideal": CIDEAL[persona], "models": {}, "pooled": {}}
        sub_persona = flat[flat["Persona"] == persona]

        # Per model
        for model in MODELS:
            result[persona]["models"][model] = {}
            for arm in ["static", "memory"]:
                sub = sub_persona[
                    (sub_persona["Model"] == model) & (sub_persona["Agent_Type"] == arm)
                ].sort_values("Seed")
                result[persona]["models"][model][arm] = {
                    metric: {
                        **agg_stats(sub[metric]),
                        "per_seed": per_seed_dict(sub, metric),
                    }
                    for metric in METRICS
                }

        # Pooled across models
        for arm in ["static", "memory"]:
            sub = sub_persona[sub_persona["Agent_Type"] == arm]
            result[persona]["pooled"][arm] = {
                metric: agg_stats(sub[metric]) for metric in METRICS
            }

        # Bidirectionality
        s = sub_persona[sub_persona["Agent_Type"] == "static"]["MAS_Deviation"]
        m = sub_persona[sub_persona["Agent_Type"] == "memory"]["MAS_Deviation"]
        delta = float(m.mean() - s.mean())
        result[persona]["bidirectionality"] = {
            "static_mean_MAS": round(float(s.mean()), 6),
            "memory_mean_MAS": round(float(m.mean()), 6),
            "delta_memory_minus_static": round(delta, 6),
            "direction": "memory HURTS (MAS up)" if delta > 0 else "memory HELPS (MAS down)",
        }

    return {
        "source": OCEAN_CSV.name,
        "scenario": "flat",
        "note": (
            "O3 personas are operationalized purely via numerical behavioral parameters "
            "(target cash fraction, risk tolerance thresholds) with no qualitative "
            "personality language. This directly addresses the reviewer prescription "
            "that 'measurable behavioral parameters would be stronger'."
        ),
        "per_persona": result,
    }


# ── 5. Three-way gap table: MBTI vs OCEAN vs O3 ───────────────────────────────
def compute_three_way_gap(df: pd.DataFrame) -> dict:
    """
    Aligns conservative and aggressive personas across three frameworks.
    MBTI: ISFJ (conservative) / ENTJ (aggressive)  -- from April claude-sonnet-4-6 flat
    OCEAN: O1_conservative / O2_aggressive          -- from results_ocean flat
    O3:   O3_conservative / O3_aggressive           -- from results_ocean flat (numerical)

    Gap = memory_MAS - static_MAS for each cell.
    """
    flat = df[df["Scenario"] == "flat"]

    # Pull April MBTI for claude-sonnet-4-6 only (to keep model constant)
    mbti_rows = []
    for arm in ["static", "memory"]:
        for persona, cideal in [("ISFJ", 1.0), ("ENTJ", 0.2)]:
            for seed in SEEDS:
                path = APRIL_BASE / f"seed{seed}" / f"{persona}_{arm}_flat_seed{seed}.csv"
                if not path.exists():
                    continue
                run_df = pd.read_csv(path)
                cash_frac = run_df["Cash"] / run_df["Portfolio_Value"]
                mas = float((cash_frac - cideal).abs().mean())
                mbti_rows.append({"Persona": persona, "Arm": arm, "Seed": seed,
                                   "MAS_Deviation": mas, "Cideal": cideal})
    mbti_df = pd.DataFrame(mbti_rows)

    def gap_entry(static_mas, memory_mas):
        delta = memory_mas - static_mas
        return {
            "static_MAS": round(static_mas, 4),
            "memory_MAS": round(memory_mas, 4),
            "gap_memory_minus_static": round(delta, 4),
            "direction": "memory HURTS (+)" if delta > 0 else "memory HELPS (-)",
        }

    table = {}
    for role, mbti_name, ocean_name, o3_name in [
        ("conservative", "ISFJ", "O1_conservative", "O3_conservative"),
        ("aggressive",   "ENTJ", "O2_aggressive",   "O3_aggressive"),
    ]:
        # MBTI (claude-sonnet-4-6, flat)
        mbti_s = mbti_df[(mbti_df.Persona == mbti_name) & (mbti_df.Arm == "static")]["MAS_Deviation"].mean()
        mbti_m = mbti_df[(mbti_df.Persona == mbti_name) & (mbti_df.Arm == "memory")]["MAS_Deviation"].mean()

        # OCEAN (claude-sonnet-4-6 only, flat, to keep model constant)
        oc_sub = flat[(flat["Persona"] == ocean_name) & (flat["Model"] == "claude-sonnet-4-6")]
        oc_s = oc_sub[oc_sub["Agent_Type"] == "static"]["MAS_Deviation"].mean()
        oc_m = oc_sub[oc_sub["Agent_Type"] == "memory"]["MAS_Deviation"].mean()

        # O3 (claude-sonnet-4-6 only, flat)
        o3_sub = flat[(flat["Persona"] == o3_name) & (flat["Model"] == "claude-sonnet-4-6")]
        o3_s = o3_sub[o3_sub["Agent_Type"] == "static"]["MAS_Deviation"].mean()
        o3_m = o3_sub[o3_sub["Agent_Type"] == "memory"]["MAS_Deviation"].mean()

        table[role] = {
            "Cideal": MBTI_CIDEAL[mbti_name],
            "MBTI": {
                "persona": mbti_name, "framework": "MBTI",
                "model": "claude-sonnet-4-6",
                **gap_entry(mbti_s, mbti_m),
            },
            "OCEAN": {
                "persona": ocean_name, "framework": "Big Five / OCEAN",
                "model": "claude-sonnet-4-6",
                **gap_entry(oc_s, oc_m),
            },
            "O3_numerical": {
                "persona": o3_name, "framework": "Numerical parameters only",
                "model": "claude-sonnet-4-6",
                **gap_entry(o3_s, o3_m),
            },
        }

    # Check if bidirectionality holds across all three frameworks
    conserv_ok = all(
        table["conservative"][fw]["gap_memory_minus_static"] < 0
        for fw in ["MBTI", "OCEAN", "O3_numerical"]
    )
    aggr_ok = all(
        table["aggressive"][fw]["gap_memory_minus_static"] > 0
        for fw in ["MBTI", "OCEAN", "O3_numerical"]
    )
    table["bidirectionality_holds_across_all_frameworks"] = conserv_ok and aggr_ok

    return {
        "note": (
            "All rows use claude-sonnet-4-6, flat scenario, 5 seeds, to keep model constant. "
            "Gap = memory_MAS - static_MAS. "
            "Conservative persona: lower gap = memory helps hold cash (good). "
            "Aggressive persona: higher gap = memory pushes away from cash (bad, "
            "but correct — mandate says stay invested)."
        ),
        "table": table,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(OCEAN_CSV)

    flat_mas = compute_ocean_flat(df)
    all_scen = compute_ocean_all_scenarios(df)
    bidir = compute_bidirectionality(df)
    o3 = compute_o3_ablation(df)
    three_way = compute_three_way_gap(df)

    summary = {
        "experiment": "OCEAN / Big Five Framework-Independence Replication",
        "source_csv": OCEAN_CSV.name,
        "models": MODELS,
        "personas": {
            "core": ["O1_conservative (Cideal=1.0)", "O2_aggressive (Cideal=0.2)"],
            "ablation": ["O3_conservative (Cideal=1.0)", "O3_aggressive (Cideal=0.2)"],
        },
        "ocean_flat_results": flat_mas,
        "ocean_all_scenarios": all_scen,
        "bidirectionality_check": bidir,
        "o3_numerical_ablation": o3,
        "three_way_gap_table": three_way,
    }

    files = {
        OUT_DIR / "ocean_flat_mas.json": flat_mas,
        OUT_DIR / "ocean_all_scenarios_mas.json": all_scen,
        OUT_DIR / "bidirectionality_check.json": bidir,
        OUT_DIR / "o3_ablation.json": o3,
        OUT_DIR / "three_way_gap_table.json": three_way,
        OUT_DIR / "summary.json": summary,
    }

    for path, data in files.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Written: {path.relative_to(PROJECT_ROOT)}")

    # ── Console report ─────────────────────────────────────────────────────────
    print("\n--- OCEAN flat MAS: memory vs static (pooled across 3 models, n=15) ---")
    for persona in ["O1_conservative", "O2_aggressive"]:
        cideal = CIDEAL[persona]
        flat_df = df[(df["Scenario"] == "flat") & (df["Persona"] == persona)]
        s = flat_df[flat_df["Agent_Type"] == "static"]["MAS_Deviation"]
        m = flat_df[flat_df["Agent_Type"] == "memory"]["MAS_Deviation"]
        print(f"  {persona} (Cideal={cideal}):  "
              f"static {s.mean():.4f}+/-{s.std():.4f}  "
              f"memory {m.mean():.4f}+/-{m.std():.4f}  "
              f"delta={m.mean()-s.mean():+.4f}")

    print("\n--- Bidirectionality check per model (flat) ---")
    for model in MODELS:
        parts = []
        for persona in ["O1_conservative", "O2_aggressive"]:
            sub = df[(df["Scenario"] == "flat") & (df["Persona"] == persona) & (df["Model"] == model)]
            s = sub[sub["Agent_Type"] == "static"]["MAS_Deviation"].mean()
            m = sub[sub["Agent_Type"] == "memory"]["MAS_Deviation"].mean()
            parts.append(f"{persona}: delta={m-s:+.4f}")
        print(f"  {model}: {' | '.join(parts)}")

    print("\n--- O3 numerical ablation (flat, pooled, n=15) ---")
    for persona in ["O3_conservative", "O3_aggressive"]:
        sub = df[(df["Scenario"] == "flat") & (df["Persona"] == persona)]
        s = sub[sub["Agent_Type"] == "static"]["MAS_Deviation"]
        m = sub[sub["Agent_Type"] == "memory"]["MAS_Deviation"]
        print(f"  {persona}:  static {s.mean():.4f}+/-{s.std():.4f}  "
              f"memory {m.mean():.4f}+/-{m.std():.4f}  "
              f"delta={m.mean()-s.mean():+.4f}")

    print("\n--- Three-way gap table (claude-sonnet-4-6, flat) ---")
    tw = three_way["table"]
    print(f"  {'Role':14}  {'Framework':22}  {'Static MAS':12}  {'Memory MAS':12}  {'Gap':8}  Direction")
    for role in ["conservative", "aggressive"]:
        for fw_key, fw_label in [("MBTI","MBTI"), ("OCEAN","OCEAN"), ("O3_numerical","O3 Numerical")]:
            e = tw[role][fw_key]
            print(f"  {role:14}  {fw_label:22}  {e['static_MAS']:<12.4f}  "
                  f"{e['memory_MAS']:<12.4f}  {e['gap_memory_minus_static']:+.4f}  "
                  f"{e['direction']}")
    bidir_ok = tw["bidirectionality_holds_across_all_frameworks"]
    print(f"\n  Bidirectionality across all 3 frameworks: {'YES' if bidir_ok else 'NO'}")


if __name__ == "__main__":
    main()
