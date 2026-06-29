#!/usr/bin/env python3
"""
Injection-Frequency Ablation Analysis
======================================

Reads the per-step CSVs produced by `run_injection_freq_ablation.py`, scores
each rationale with the persona classifier, and produces:

  - injection_freq_curve.csv     — mean P(intended) per (k, scenario, persona)
  - injection_freq_curve.png     — Pareto curve: token-overhead vs adherence
  - injection_freq_table.csv     — per-k effect sizes vs the static (k=∞) baseline

Inputs
------
  results/injection_freq/                                      ← per-step CSVs
  analysis/outputs/persona_classifier/model/                   ← trained classifier

Outputs
-------
  analysis/outputs/injection_freq/

Usage
-----
    python analysis/injection_freq_analysis.py
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PERSONAS      = ["ENTJ", "ISFJ", "INTJ"]
PERSONA_TO_ID = {p: i for i, p in enumerate(PERSONAS)}
MAX_LEN       = 256
INF_TAG       = "INF"


# ──────────────────────────────────────────────────────────────────────────
#  Load + score rationales
# ──────────────────────────────────────────────────────────────────────────

def load_per_step(results_dir: Path):
    csvs = list(results_dir.rglob("*.csv"))
    csvs = [c for c in csvs if "master_summary" not in c.name]
    if not csvs:
        raise FileNotFoundError(f"No per-step CSVs under {results_dir}")
    dfs = []
    for f in csvs:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        # Derive integer day
        if "Date" in d.columns and "Day" not in d.columns:
            d["Day"] = pd.to_numeric(
                d["Date"].astype(str).str.extract(r"Day-(\d+)", expand=False),
                errors="coerce"
            ).astype("Int64")
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def score_rationales(df: pd.DataFrame, model_dir: Path):
    """Apply the trained persona classifier."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if "Persona" not in df.columns and "MBTI" in df.columns:
        df = df.rename(columns={"MBTI": "Persona"})

    df = df[df["Persona"].isin(PERSONAS)].copy()
    df = df[df["Rationale"].notna()
            & (df["Rationale"].astype(str).str.strip().str.len() >= 5)]
    df["Rationale"] = df["Rationale"].astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    probs = np.zeros((len(df), len(PERSONAS)), dtype=np.float32)
    batch = 64
    texts = df["Rationale"].tolist()
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            chunk = texts[s:s+batch]
            enc = tok(chunk, truncation=True, max_length=MAX_LEN,
                       padding=True, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            probs[s:s+batch] = torch.softmax(logits, dim=-1).cpu().numpy()

    intended_idx = df["Persona"].map(PERSONA_TO_ID).astype(int).to_numpy()
    df = df.reset_index(drop=True)
    df["p_intended"] = probs[np.arange(len(df)), intended_idx]
    df["correct"]    = (probs.argmax(axis=1) == intended_idx).astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────
#  Aggregate + plot
# ──────────────────────────────────────────────────────────────────────────

def parse_k(tag) -> float:
    s = str(tag).strip()
    return math.inf if s.upper() == INF_TAG else float(s)


def aggregate_curve(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean adherence per (k, scenario, persona)."""
    if "Injection_Frequency" not in scored.columns:
        raise ValueError("Per-step CSVs missing 'Injection_Frequency' column. "
                         "Did you run run_injection_freq_ablation.py?")
    g = (scored
         .groupby(["Injection_Frequency", "Scenario", "Persona"])
         ["p_intended"].agg(["mean", "std", "count"])
         .reset_index())
    g["k_numeric"] = g["Injection_Frequency"].map(parse_k)
    g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    g["ci_lo"] = g["mean"] - 1.96 * g["se"]
    g["ci_hi"] = g["mean"] + 1.96 * g["se"]
    return g.sort_values(["Scenario", "Persona", "k_numeric"])


def plot_pareto(curve: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size":      13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))

    persona_colors = {"ENTJ": "#E11D48", "ISFJ": "#10B981", "INTJ": "#3B82F6"}

    # X-axis: 1/k (refresh rate, in "% of turns reminded"). Static (k=∞) -> 0.
    for persona in PERSONAS:
        sub = curve[curve["Persona"] == persona].copy()
        if sub.empty:
            continue
        # Average across scenarios for a cleaner curve
        agg = sub.groupby("k_numeric").agg(
            mean=("mean", "mean"),
            ci_lo=("ci_lo", "mean"),
            ci_hi=("ci_hi", "mean"),
        ).reset_index().sort_values("k_numeric")

        # Convert k -> refresh rate. Cap inf at the rightmost log-tick.
        refresh = np.where(np.isinf(agg["k_numeric"]), 0.0, 1.0 / agg["k_numeric"]) * 100

        order = np.argsort(refresh)
        x = refresh[order]
        y = agg["mean"].values[order]
        lo = agg["ci_lo"].values[order]
        hi = agg["ci_hi"].values[order]

        ax.plot(x, y, marker="o", markersize=10, linewidth=2.5,
                color=persona_colors[persona], label=persona)
        ax.fill_between(x, lo, hi, color=persona_colors[persona], alpha=0.15)

        # Annotate each point with the k value
        for k_val, xi, yi in zip(agg["k_numeric"].values[order], x, y):
            k_txt = "INF" if math.isinf(k_val) else f"k={int(k_val)}"
            ax.annotate(k_txt, (xi, yi), textcoords="offset points",
                         xytext=(6, 6), fontsize=10, color=persona_colors[persona])

    ax.set_xlabel("Mandate refresh rate  (% of turns reminded)")
    ax.set_ylabel("P(intended persona)")
    ax.set_title("Injection-frequency Pareto: token cost vs persona adherence",
                  fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Persona", fontsize=12, frameon=True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Plot] Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results/injection_freq")
    p.add_argument("--classifier",  default="analysis/outputs/persona_classifier/model")
    p.add_argument("--output-dir",  default="analysis/outputs/injection_freq")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    classifier  = Path(args.classifier)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not classifier.exists():
        print(f"ERROR: trained classifier not at {classifier}")
        print("Run `python analysis/persona_classifier.py train` first.")
        return 2

    print(f"[Load] Reading per-step CSVs under {results_dir} ...")
    raw = load_per_step(results_dir)
    print(f"[Load] {len(raw):,} rationale-level rows.")

    print(f"[Score] Applying classifier from {classifier} ...")
    scored = score_rationales(raw, classifier)
    print(f"[Score] {len(scored):,} rows scored.")

    print(f"[Aggregate] Building (k × scenario × persona) curve ...")
    curve = aggregate_curve(scored)
    curve_path = out_dir / "injection_freq_curve.csv"
    curve.to_csv(curve_path, index=False)
    print(f"[Aggregate] {len(curve)} rows -> {curve_path}")

    # Print headline table: mean P(intended) per k (averaged over personas / scenarios)
    head = curve.groupby("k_numeric")["mean"].agg(["mean", "std"]).reset_index()
    head["k"] = head["k_numeric"].apply(lambda v: "INF" if math.isinf(v) else int(v))
    head = head[["k", "mean", "std"]].rename(columns={"mean": "Adherence_mean", "std": "Adherence_std"})
    print("\nHeadline curve (averaged across personas & scenarios):")
    print(head.round(4).to_string(index=False))

    plot_pareto(curve, out_dir / "injection_freq_curve.png")

    # Save scored rationales too, in case other analyses want it
    scored.to_csv(out_dir / "scored_rationales.csv", index=False)
    print(f"\n[Done] All outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
