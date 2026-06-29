# -*- coding: utf-8 -*-
"""
T Calibration Summary - compact tabular view of t_calibration results.
Outputs two tables:
  1. Mean key metrics per (Model, Scenario, T_Value)
  2. Wilcoxon p-values per (Model, Scenario) across T values (with significance markers)
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
MASTER = ROOT / "analysis/outputs/t_calibration_outputs/t_calibration_master.csv"
WILCOXON = ROOT / "analysis/outputs/t_calibration_outputs/t_calibration_wilcoxon.csv"

master = pd.read_csv(MASTER)
wilcox = pd.read_csv(WILCOXON)

# ── Table 1: Mean performance metrics per model / scenario / T ───────────────
agg = (
    master.groupby(["Model", "Scenario", "T_Value"])
    .agg(
        Return_Pct=("Return_Pct", "mean"),
        Max_DD=("Max_Drawdown_Pct", "mean"),
        Rationality=("Rationality_Score", "mean"),
        MAS_Dev=("MAS_Deviation", "mean"),
        N=("Return_Pct", "count"),
    )
    .reset_index()
)

# Shorten model names for display
model_short = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-2.5-flash": "Gemini-2.5",
    "claude-sonnet-4-6": "Claude-S-4.6",
}
agg["Model"] = agg["Model"].map(model_short).fillna(agg["Model"])
wilcox["Model"] = wilcox["Model"].map(model_short).fillna(wilcox["Model"])

print("=" * 80)
print("TABLE 1 - Mean Metrics by Model / Scenario / T")
print("=" * 80)

for scenario in ["flat", "bull_trap", "crash"]:
    sub = agg[agg["Scenario"] == scenario].copy()
    print(f"\n  Scenario: {scenario.upper()}")
    print(
        f"  {'Model':<15} {'T':>4}  {'Ret%':>7}  {'MaxDD%':>8}  {'Rational':>9}  {'MAS_Dev':>8}  {'N':>3}"
    )
    print("  " + "-" * 65)
    for _, row in sub.sort_values(["Model", "T_Value"]).iterrows():
        print(
            f"  {row['Model']:<15} {int(row['T_Value']):>4}  "
            f"{row['Return_Pct']:>7.2f}  {row['Max_DD']:>8.2f}  "
            f"{row['Rationality']:>9.1f}  {row['MAS_Dev']:>8.4f}  {int(row['N']):>3}"
        )

# ── Table 2: Wilcoxon p-values — metric differs by scenario ─────────────────
# Each scenario has its own primary metric; pivot T as columns
print("\n")
print("=" * 80)
print("TABLE 2 - Wilcoxon p-values (static vs memory)  * = p<0.05")
print("           flat->MAS_Deviation | bull_trap->Rationality | crash->MaxDrawdown")
print("=" * 80)

T_VALUES = sorted(wilcox["T_Value"].unique())

# Pivot: index=(Model,Scenario,Metric), columns=T_Value
piv = wilcox.pivot_table(
    index=["Model", "Scenario", "Metric"],
    columns="T_Value",
    values="P_Value",
    aggfunc="first",
)
sig = wilcox.pivot_table(
    index=["Model", "Scenario", "Metric"],
    columns="T_Value",
    values="Significant",
    aggfunc="first",
)

header = f"  {'Model':<15} {'Scenario':<12} {'Metric':<22}" + "".join(
    f"  T={t:>3}" for t in T_VALUES
)
print(header)
print("  " + "-" * (len(header) - 2))

for (model, scenario, metric), row in piv.iterrows():
    cells = []
    for t in T_VALUES:
        p = row.get(t, float("nan"))
        s = sig.loc[(model, scenario, metric), t] if not pd.isna(p) else False
        marker = "*" if s else " "
        cells.append(f"{p:>5.3f}{marker}")
    print(f"  {model:<15} {scenario:<12} {metric:<22}" + "  ".join(cells))

# ── Table 3: First T where significance is reached ───────────────────────────
print("\n")
print("=" * 80)
print("TABLE 3 - First T achieving p<0.05 significance per model/scenario")
print("=" * 80)
print(f"  {'Model':<15} {'Scenario':<12} {'Metric':<22}  First sig. T")
print("  " + "-" * 60)

for (model, scenario, metric), row in sig.iterrows():
    first_sig = next((t for t in T_VALUES if row.get(t) == True), None)
    marker = str(first_sig) if first_sig else "--  (none)"
    print(f"  {model:<15} {scenario:<12} {metric:<22}  {marker}")
