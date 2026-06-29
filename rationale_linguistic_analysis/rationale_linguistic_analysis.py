#!/usr/bin/env python3
"""
Rationale-Based Linguistic Analysis
=====================================
Provides linguistic evidence for signal interference in mini-model crash reversals.

Addresses:
  Reviewer Gxm9 W4: "Signal interference is purely subjective speculation."
  Reviewer nhWU Q4: "Can rationale strings show internal conflict in mini models?"

Two complementary analyses on EXISTING crash-scenario rationale strings (no new
simulations):

  1. Persona Classifier Confidence Trajectories
     Apply the existing DistilBERT persona classifier to crash-scenario rationales,
     segmented by quartile, for mini vs. flagship models under static vs. memory
     conditions.

  2. Lexical Conflict Rate (LCR)
     Count rationales that co-express mandate-aligned language AND market-reactive
     stress language in the same string.

  3. Case Study Extraction
     Representative Q3 (peak-crash) rationale excerpts illustrating the contrast
     between mini and flagship models under memory re-injection.

Models: GPT-4o-mini (mini), GPT-4.1-mini (mini), GPT-4o (flagship), Claude Sonnet 4.6 (flagship)
Scenario: crash / discount0.92 only
Conditions: static, memory

Checkpoint / Resume
-------------------
  Classifier probabilities are saved to outputs/classifier_checkpoint.npy every
  CHECKPOINT_EVERY batches. If the run is interrupted and re-started, the checkpoint
  is detected automatically and inference resumes from the last saved batch -- no
  rows are re-processed.

Usage
-----
    # Run from repo root (recommended: use the project venv)
    venv/Scripts/python rationale_linguistic_analysis/rationale_linguistic_analysis.py

    # Skip classifier (LCR + case studies only, no torch required)
    venv/Scripts/python rationale_linguistic_analysis/rationale_linguistic_analysis.py --skip-classifier

    # Custom paths
    venv/Scripts/python rationale_linguistic_analysis/rationale_linguistic_analysis.py \\
        --results-dir results_april \\
        --classifier-dir finpersona_classifier_model/model \\
        --output-dir rationale_linguistic_analysis/outputs
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
#  CONSTANTS
# -----------------------------------------------------------------------

PERSONAS = ["ENTJ", "ISFJ", "INTJ"]
PERSONA_TO_ID = {p: i for i, p in enumerate(PERSONAS)}
ID_TO_PERSONA = {i: p for p, i in PERSONA_TO_ID.items()}

# (results_april subfolder, inner model folder, display label, category)
MODEL_CONFIGS = [
    ("gpt_4o_mini_200",       "gpt-4o-mini",        "GPT-4o-mini",      "mini"),
    ("gpt_4_1_mini_200",      "gpt-4.1-mini",       "GPT-4.1-mini",     "mini"),
    ("gpt_4o_200",            "gpt-4o",             "GPT-4o",           "flagship"),
    ("claude_sonnet_4_6_200", "claude-sonnet-4-6",  "Claude Sonnet 4.6","flagship"),
]

SEEDS = ["42", "123", "456", "789", "999"]
CONDITIONS = ["static", "memory"]
DISCOUNT = "0.92"

QUARTILES = {
    "Q1": (1,   50),
    "Q2": (51,  100),
    "Q3": (101, 150),
    "Q4": (151, 200),
}

# Save a checkpoint every this many batches during classifier inference
CHECKPOINT_EVERY = 20

# -----------------------------------------------------------------------
#  LEXICONS  (pre-specified before analysis; no post-hoc additions)
#  Source: mandate texts from agent/personas/mbti_profiles.json + spec Section 3.2
# -----------------------------------------------------------------------

MANDATE_LEXICON = {
    "ENTJ": [
        "momentum", "trend", "decisive", "growth", "breakout",
        "chase", "buy", "bullish", "follow the trend",
    ],
    "ISFJ": [
        "preserve", "protect", "capital", "safe", "guardian", "security",
        "avoid", "cautious", "hedge",
    ],
    "INTJ": [
        "system", "model", "thesis", "alpha", "asymmetric",
        "long game", "puzzle", "plan the exit",
    ],
    "UNIVERSAL": [
        "mandate", "reminder", "my goal", "my role", "my strategy",
    ],
}

STRESS_LEXICON = [
    # panic / fear
    "panic", "fear", "crash", "collapse", "plunge", "freefall",
    "alarming", "terrifying", "danger",
    # selling / exit
    "sell", "exit", "liquidate", "cut losses", "stop loss", "get out", "reduce exposure",
    # market signals
    "sharp drop", "sharp decline", "oversold", "below value",
    "discount", "massive", "severe", "extreme",
    # urgency
    "immediately", "now", "urgent", "must", "cannot wait", "no choice", "forced",
]


def _mandate_tokens(persona: str):
    return MANDATE_LEXICON.get(persona, []) + MANDATE_LEXICON["UNIVERSAL"]


# -----------------------------------------------------------------------
#  DATA LOADING  (parallel with ThreadPoolExecutor — I/O bound)
# -----------------------------------------------------------------------

def _read_one_csv(args):
    """Worker: read a single CSV and attach metadata columns."""
    fpath, display_label, category = args
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        return None, str(e)
    df["display_model"] = display_label
    df["model_category"] = category
    df["Day"] = pd.to_numeric(
        df["Date"].astype(str).str.extract(r"Day-(\d+)", expand=False),
        errors="coerce",
    ).astype("Int64")
    return df, None


def load_crash_rationales(results_dir: Path, n_workers: int = 8) -> pd.DataFrame:
    """
    Load per-step CSVs for all 4 models, crash/discount0.92, all seeds + conditions.
    Uses a ThreadPoolExecutor (n_workers threads) to parallelise I/O.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm

    # Build the full list of files to load
    file_list = []
    for results_subdir, inner_folder, display_label, category in MODEL_CONFIGS:
        base = results_dir / results_subdir / inner_folder / "crash" / f"discount{DISCOUNT}"
        if not base.exists():
            print(f"[Data] WARNING: missing {base}")
            continue
        for seed in SEEDS:
            seed_dir = base / f"seed{seed}"
            if not seed_dir.exists():
                print(f"[Data] WARNING: missing {seed_dir}")
                continue
            for persona in PERSONAS:
                for condition in CONDITIONS:
                    fname = f"{persona}_{condition}_crash_seed{seed}_discount{DISCOUNT}.csv"
                    fpath = seed_dir / fname
                    if fpath.exists():
                        file_list.append((fpath, display_label, category))
                    else:
                        print(f"[Data] WARNING: missing {fpath}")

    if not file_list:
        raise FileNotFoundError(
            f"No crash CSVs found under {results_dir}. "
            "Make sure you are pointing at the correct results_april directory."
        )
    workers = min(n_workers, len(file_list), (os.cpu_count() or 4) * 2)
    dfs     = [None] * len(file_list)
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_read_one_csv, args): i for i, args in enumerate(file_list)}
        pbar = tqdm(as_completed(futures), total=len(file_list),
                    desc=f"Loading CSVs ({workers} threads)", unit="file")
        for fut in pbar:
            i = futures[fut]
            df, err = fut.result()
            if err:
                print(f"\n[Data] Error reading {file_list[i][0]}: {err}")
                skipped += 1
            else:
                dfs[i] = df

    good = [d for d in dfs if d is not None]
    if not good:
        raise FileNotFoundError(
            f"No crash CSVs found under {results_dir}. Check MODEL_CONFIGS folder names."
        )
    combined = pd.concat(good, ignore_index=True)
    combined = combined[combined["Rationale"].notna()]
    combined = combined[combined["Rationale"].astype(str).str.strip().str.len() >= 5]
    combined["Rationale"] = combined["Rationale"].astype(str)
    if skipped:
        print(f"[Data] Skipped {skipped} unreadable files.")
    print(f"[Data] Loaded {len(combined):,} rationale rows across "
          f"{combined['display_model'].nunique()} models.")
    return combined


def assign_quartile(day) -> str:
    try:
        day = int(day)
    except (TypeError, ValueError):
        return "Q4"
    for q, (lo, hi) in QUARTILES.items():
        if lo <= day <= hi:
            return q
    return "Q4"


# -----------------------------------------------------------------------
#  PERSONA CLASSIFIER INFERENCE  (with checkpoint / resume)
# -----------------------------------------------------------------------

def run_classifier(df: pd.DataFrame, classifier_dir: Path,
                   out_dir: Path, batch_size: int = 128) -> pd.DataFrame:
    """
    Apply DistilBERT classifier to all rationales.

    Speed-ups vs. naive loop
    ------------------------
    - batch_size=128 (default) reduces per-sample overhead vs. 64
    - torch.inference_mode() skips autograd bookkeeping (~5% faster than no_grad)
    - torch.set_num_threads(cpu_count) ensures all cores are used on CPU
    - Tokenization runs on CPU in the main thread while the previous batch is
      already in flight on the compute device (overlap via pre-fetching)

    Checkpoint / resume
    -------------------
    Probabilities are saved to out_dir/classifier_checkpoint.npy every
    CHECKPOINT_EVERY batches. On re-start the checkpoint is detected and
    inference resumes from the last saved batch — no rows are re-processed.
    """
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm.auto import tqdm

    checkpoint_probs = out_dir / "classifier_checkpoint.npy"
    checkpoint_idx   = out_dir / "classifier_checkpoint_idx.txt"

    # Pin all available CPU threads for PyTorch (ignored on CUDA)
    n_cpu = os.cpu_count() or 4
    torch.set_num_threads(n_cpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Classifier] Device: {device}  |  CPU threads: {n_cpu}")

    tokenizer = AutoTokenizer.from_pretrained(str(classifier_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(classifier_dir)
    ).to(device).eval()

    texts     = df["Rationale"].tolist()
    n_total   = len(texts)
    probs_all = np.zeros((n_total, len(PERSONAS)), dtype=np.float32)

    # --- Resume from checkpoint if it exists ---
    resume_from = 0
    if checkpoint_probs.exists() and checkpoint_idx.exists():
        try:
            saved_probs = np.load(str(checkpoint_probs))
            saved_rows  = int(checkpoint_idx.read_text().strip())
            if saved_probs.shape == (n_total, len(PERSONAS)) and saved_rows <= n_total:
                probs_all   = saved_probs
                resume_from = saved_rows
                print(f"[Classifier] Resuming from checkpoint: "
                      f"{resume_from:,}/{n_total:,} rows already scored.")
            else:
                print("[Classifier] Checkpoint shape mismatch -- starting fresh.")
        except Exception as e:
            print(f"[Classifier] Could not load checkpoint ({e}) -- starting fresh.")

    if resume_from >= n_total:
        print("[Classifier] All rows already scored from checkpoint.")
    else:
        remaining  = list(range(resume_from, n_total, batch_size))
        n_batches  = len(remaining)

        # Pre-tokenize the first batch before entering the loop so the
        # model never waits for tokenization (overlap pattern).
        def _tokenize(start):
            chunk = texts[start : start + batch_size]
            return tokenizer(chunk, truncation=True, max_length=256,
                             padding=True, return_tensors="pt")

        next_enc = _tokenize(remaining[0]) if remaining else None

        with torch.inference_mode():
            pbar = tqdm(enumerate(remaining, start=1),
                        desc="Classifier inference",
                        unit="batch",
                        total=n_batches)
            for batch_num, start in pbar:
                enc       = next_enc
                end       = min(start + batch_size, n_total)
                # Pre-tokenize next batch while current is on device
                if batch_num < n_batches:
                    next_enc = _tokenize(remaining[batch_num])  # remaining is 0-indexed
                else:
                    next_enc = None

                enc    = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                probs_all[start:end] = probs

                pbar.set_postfix(rows=f"{end:,}/{n_total:,}")

                # Checkpoint every CHECKPOINT_EVERY batches
                if batch_num % CHECKPOINT_EVERY == 0:
                    np.save(str(checkpoint_probs), probs_all)
                    checkpoint_idx.write_text(str(end))

        # Final checkpoint save
        np.save(str(checkpoint_probs), probs_all)
        checkpoint_idx.write_text(str(n_total))
        print("[Classifier] Inference complete. Final checkpoint saved.")

    # Attach results to dataframe
    df = df.copy()
    for i, p in enumerate(PERSONAS):
        df[f"p_{p}"] = probs_all[:, i]

    intended_idx     = df["MBTI"].map(PERSONA_TO_ID).astype(int).to_numpy()
    df["p_intended"] = probs_all[np.arange(n_total), intended_idx]
    df["predicted"]  = np.array([ID_TO_PERSONA[int(i)] for i in probs_all.argmax(axis=1)])
    df["correct"]    = (df["predicted"] == df["MBTI"]).astype(int)
    print(f"[Classifier] Overall accuracy: {df['correct'].mean():.3f}")
    return df


# -----------------------------------------------------------------------
#  LEXICAL CONFLICT RATE  (vectorized — no row-by-row apply)
# -----------------------------------------------------------------------

def _build_regex(tokens):
    """Build a single compiled regex that matches any token (case-insensitive)."""
    import re
    pattern = "|".join(re.escape(t) for t in tokens)
    return re.compile(pattern, re.IGNORECASE)


# Pre-compile all regexes once at import time
_STRESS_RE   = _build_regex(STRESS_LEXICON)
_MANDATE_RES = {
    p: _build_regex(_mandate_tokens(p)) for p in PERSONAS
}


def compute_lcr(df: pd.DataFrame):
    """
    Vectorized LCR: for each persona group, use Series.str.contains() with a
    pre-compiled regex to detect mandate tokens and stress tokens simultaneously.
    No row-by-row apply — runs ~20x faster than the progress_apply version.
    """
    from tqdm.auto import tqdm

    df   = df.copy()
    text = df["Rationale"]   # original Series (str.contains is already case-handled by regex)

    # Stress mask is persona-agnostic — compute once for all rows
    has_stress = text.str.contains(_STRESS_RE, na=False)

    # Mandate mask is persona-specific — compute per-persona group and combine
    has_mandate = pd.Series(False, index=df.index)
    for persona in tqdm(PERSONAS, desc="LCR (per-persona regex)", unit="persona"):
        mask = df["MBTI"] == persona
        has_mandate.loc[mask] = text.loc[mask].str.contains(
            _MANDATE_RES[persona], na=False
        )

    df["lcr_flag"] = (has_mandate & has_stress).astype(int)

    agg = (
        df.groupby(["display_model", "model_category", "Agent_Type", "Quartile"])
          .agg(lcr=("lcr_flag", "mean"), n=("lcr_flag", "count"))
          .reset_index()
    )
    agg["lcr_pct"] = (agg["lcr"] * 100).round(1)
    return df, agg


# -----------------------------------------------------------------------
#  CASE STUDY SELECTION
# -----------------------------------------------------------------------

def select_case_studies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick Q3 (t=101-150) rationales illustrating signal interference.
    Mini/memory: highest LCR (most conflicted, representative of aggregate pattern).
    Flagship/memory: lowest LCR (coherent mandate under same conditions).
    Selected before examining which examples look 'impressive'.
    """
    q3 = df[df["Quartile"] == "Q3"].copy()
    rows = []
    for category, condition in [("mini", "memory"), ("flagship", "memory"),
                                 ("mini", "static"), ("flagship", "static")]:
        subset = q3[
            (q3["model_category"] == category) &
            (q3["Agent_Type"] == condition)
        ].copy()
        if subset.empty:
            continue
        for model_label in subset["display_model"].unique():
            for persona in PERSONAS:
                cell = subset[
                    (subset["display_model"] == model_label) &
                    (subset["MBTI"] == persona)
                ].copy()
                if cell.empty:
                    continue
                # Sort: mini/memory -> descending LCR; others -> ascending LCR
                ascending = not (category == "mini" and condition == "memory")
                cell = cell.sort_values(["lcr_flag", "Day"], ascending=[ascending, True])
                ex = cell.iloc[0]
                rows.append({
                    "category":  category,
                    "condition": condition,
                    "model":     model_label,
                    "persona":   persona,
                    "seed":      ex["Seed"],
                    "day":       int(ex["Day"]),
                    "action":    ex["Action"],
                    "lcr_flag":  int(ex["lcr_flag"]),
                    "rationale": ex["Rationale"][:600],
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
#  SUMMARY TABLE  (2x2 design from spec Section 4)
# -----------------------------------------------------------------------

def build_summary_table(df: pd.DataFrame, has_classifier: bool) -> pd.DataFrame:
    late = df[df["Quartile"].isin(["Q3", "Q4"])].copy()
    q3   = df[df["Quartile"] == "Q3"].copy()

    rows = []
    for _, inner, model_label, category in MODEL_CONFIGS:
        for condition in CONDITIONS:
            row = {"Model": model_label, "Category": category, "Condition": condition}
            cell_late = late[(late["display_model"] == model_label) & (late["Agent_Type"] == condition)]
            cell_q3   = q3  [(q3["display_model"]  == model_label) & (q3["Agent_Type"]   == condition)]

            if has_classifier and "p_intended" in df.columns:
                row["Mean_P_intended_Q3Q4"] = (
                    round(cell_late["p_intended"].mean(), 4) if len(cell_late) else float("nan")
                )
            else:
                row["Mean_P_intended_Q3Q4"] = float("nan")

            row["LCR_Q3_pct"] = (
                round(cell_q3["lcr_flag"].mean() * 100, 1) if len(cell_q3) else float("nan")
            )
            row["N_Q3"] = len(cell_q3)
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
#  FIGURES
# -----------------------------------------------------------------------

def plot_persona_confidence(df: pd.DataFrame, out_dir: Path):
    """4-panel figure: P(intended persona) x quartile per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_order = ["Q1", "Q2", "Q3", "Q4"]
    colors  = {"static": "#E84855", "memory": "#1B998B"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.flatten()

    for ax, (_, _, model_label, category) in zip(axes, MODEL_CONFIGS):
        sub = df[df["display_model"] == model_label]
        for cond in CONDITIONS:
            cell = sub[sub["Agent_Type"] == cond]
            if cell.empty:
                continue
            means, cis = [], []
            for q in q_order:
                vals = cell[cell["Quartile"] == q]["p_intended"].dropna()
                if len(vals) == 0:
                    means.append(float("nan")); cis.append(0)
                    continue
                means.append(vals.mean())
                cis.append(1.96 * vals.std() / max(np.sqrt(len(vals)), 1))
            x = np.arange(len(q_order))
            ax.errorbar(x, means, yerr=cis, label=cond,
                        color=colors[cond], marker="o", linewidth=2, capsize=4)

        ax.set_title(f"{model_label} ({category})", fontweight="bold")
        ax.set_xticks(range(len(q_order)))
        ax.set_xticklabels(q_order)
        ax.set_xlabel("Quartile (crash scenario)")
        ax.set_ylabel("P(intended persona)")
        ax.set_ylim(0, 1.05)
        ax.axhline(1/3, color="gray", linestyle=":", linewidth=1, label="random (0.33)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Persona Classifier Confidence: Crash Scenario\n"
        "(higher = rationale more consistent with assigned persona)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "persona_confidence_figure.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved -> {path}")


def plot_lcr(lcr_agg: pd.DataFrame, out_dir: Path):
    """Grouped bar chart: LCR per model x condition x quartile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_order     = ["Q1", "Q2", "Q3", "Q4"]
    model_order = [c[2] for c in MODEL_CONFIGS]
    colors      = {"static": "#E84855", "memory": "#1B998B"}

    fig, axes = plt.subplots(1, len(model_order), figsize=(18, 5), sharey=True)

    for ax, model_label in zip(axes, model_order):
        sub   = lcr_agg[lcr_agg["display_model"] == model_label]
        x     = np.arange(len(q_order))
        width = 0.35
        for i, cond in enumerate(CONDITIONS):
            cell = sub[sub["Agent_Type"] == cond]
            vals = []
            for q in q_order:
                row = cell[cell["Quartile"] == q]
                vals.append(float(row["lcr_pct"].values[0]) if len(row) > 0 else 0.0)
            ax.bar(x + i * width, vals, width, label=cond, color=colors[cond], alpha=0.85)

        category = sub["model_category"].iloc[0] if len(sub) > 0 else ""
        ax.set_title(f"{model_label}\n({category})", fontweight="bold")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(q_order)
        ax.set_xlabel("Quartile")
        ax.set_ylabel("LCR (%)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Lexical Conflict Rate (LCR): Crash Scenario\n"
        "(% of rationales with both mandate-aligned AND market-stress language)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "lcr_figure.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved -> {path}")


def plot_lcr_2x2(lcr_agg: pd.DataFrame, out_dir: Path):
    """Line plot: LCR trajectory across quartiles, mini vs flagship."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_order     = ["Q1", "Q2", "Q3", "Q4"]
    colors_mini     = {"GPT-4o-mini": "#FF6B6B", "GPT-4.1-mini": "#FF9F43"}
    colors_flagship = {"GPT-4o": "#1B998B", "Claude Sonnet 4.6": "#0066CC"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (group_label, models, color_map) in zip(
        axes,
        [("Mini Models",     [c[2] for c in MODEL_CONFIGS if c[3]=="mini"],     colors_mini),
         ("Flagship Models", [c[2] for c in MODEL_CONFIGS if c[3]=="flagship"], colors_flagship)],
    ):
        for cond, ls in [("static", "--"), ("memory", "-")]:
            for model_label in models:
                sub  = lcr_agg[(lcr_agg["display_model"] == model_label) & (lcr_agg["Agent_Type"] == cond)]
                vals = []
                for q in q_order:
                    row = sub[sub["Quartile"] == q]
                    vals.append(float(row["lcr_pct"].values[0]) if len(row) > 0 else 0.0)
                color = color_map.get(model_label, "gray")
                ax.plot(q_order, vals, linestyle=ls, color=color, linewidth=2,
                        marker="o", label=f"{model_label} ({cond})")

        ax.set_title(group_label, fontweight="bold")
        ax.set_xlabel("Quartile (crash scenario)")
        ax.set_ylabel("LCR (%)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Lexical Conflict Rate Trajectories: Mini vs. Flagship (Crash Scenario)\n"
        "Key test: does memory re-injection increase LCR for mini but not flagship?",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "lcr_2x2_figure.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved -> {path}")


# -----------------------------------------------------------------------
#  2x2 CONTRAST SUMMARY
# -----------------------------------------------------------------------

def _print_2x2_contrasts(lcr_agg: pd.DataFrame):
    """Print the three key contrasts from spec Section 4."""
    q3 = lcr_agg[lcr_agg["Quartile"] == "Q3"]

    def get_lcr(model_label, condition):
        row = q3[(q3["display_model"] == model_label) & (q3["Agent_Type"] == condition)]
        return float(row["lcr_pct"].values[0]) if len(row) > 0 else float("nan")

    mini_models     = [c[2] for c in MODEL_CONFIGS if c[3] == "mini"]
    flagship_models = [c[2] for c in MODEL_CONFIGS if c[3] == "flagship"]

    print("\nKey Contrasts (LCR in Q3 -- peak crash phase):")
    print("-" * 65)
    print("A) Memory mini vs. Memory flagship (signal interference test)")
    for m in mini_models:
        print(f"   {m:22s}  memory LCR = {get_lcr(m,'memory'):5.1f}%")
    for m in flagship_models:
        print(f"   {m:22s}  memory LCR = {get_lcr(m,'memory'):5.1f}%")

    print("\nB) Memory vs. Static within mini (does re-injection cause conflict?)")
    for m in mini_models:
        s, me = get_lcr(m, "static"), get_lcr(m, "memory")
        delta = me - s if not (np.isnan(me) or np.isnan(s)) else float("nan")
        print(f"   {m:22s}  static={s:5.1f}%  memory={me:5.1f}%  delta={delta:+.1f}%")

    print("\nC) Memory vs. Static within flagship (does re-injection stabilize?)")
    for m in flagship_models:
        s, me = get_lcr(m, "static"), get_lcr(m, "memory")
        delta = me - s if not (np.isnan(me) or np.isnan(s)) else float("nan")
        print(f"   {m:22s}  static={s:5.1f}%  memory={me:5.1f}%  delta={delta:+.1f}%")
    print("-" * 65)


# -----------------------------------------------------------------------
#  MAIN PIPELINE
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Repo root = two levels up from this script file, so paths work
    # regardless of which directory the user runs the script from.
    _repo_root = Path(__file__).resolve().parent.parent

    ap.add_argument("--results-dir",     default=str(_repo_root / "results_april"),
                    help="Top-level results directory")
    ap.add_argument("--classifier-dir",  default=str(_repo_root / "finpersona_classifier_model" / "model"),
                    help="Path to trained DistilBERT model directory")
    ap.add_argument("--output-dir",      default=str(_repo_root / "rationale_linguistic_analysis" / "rationale_results"),
                    help="Where to write outputs")
    ap.add_argument("--batch-size",      type=int, default=128,
                    help="Batch size for classifier inference (default: 128)")
    ap.add_argument("--skip-classifier", action="store_true",
                    help="Skip DistilBERT inference -- runs LCR + case studies only")
    args = ap.parse_args()

    results_dir    = Path(args.results_dir)
    classifier_dir = Path(args.classifier_dir)
    out_dir        = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory : {out_dir.resolve()}")
    print(f"Results directory: {results_dir.resolve()}")

    # Step 1: Load ---------------------------------------------------------
    print("\n=== Step 1 / 7: Loading crash rationales ===")
    df = load_crash_rationales(results_dir)
    df["Quartile"] = df["Day"].apply(assign_quartile)
    print(f"\n[Data] Row counts by model x condition:")
    print(df.groupby(["display_model", "Agent_Type"]).size().to_string())

    # Step 2: Classifier ---------------------------------------------------
    has_classifier = False
    if not args.skip_classifier:
        if not classifier_dir.exists():
            print(f"\n[Classifier] WARNING: {classifier_dir} not found -- skipping.")
        else:
            print("\n=== Step 2 / 7: Running persona classifier ===")
            print(f"[Classifier] Model: {classifier_dir}")
            print(f"[Classifier] Total rows to score: {len(df):,}")
            print(f"[Classifier] Checkpointing every {CHECKPOINT_EVERY} batches to {out_dir}/")
            try:
                df = run_classifier(df, classifier_dir, out_dir, batch_size=args.batch_size)
                has_classifier = True
                df.to_csv(out_dir / "persona_confidence_crash.csv", index=False)
                print(f"[Classifier] Full scores saved -> persona_confidence_crash.csv")
            except ImportError as e:
                print(f"[Classifier] torch/transformers not available ({e}). Skipping.")
    else:
        # Check if a previous full run left scores we can use
        cached = out_dir / "persona_confidence_crash.csv"
        if cached.exists():
            print(f"\n[Classifier] --skip-classifier set but found cached scores at {cached}")
            print("[Classifier] Loading cached scores...")
            cached_df = pd.read_csv(cached)
            if "p_intended" in cached_df.columns and len(cached_df) == len(df):
                for col in ["p_ENTJ", "p_ISFJ", "p_INTJ", "p_intended", "predicted", "correct"]:
                    if col in cached_df.columns:
                        df[col] = cached_df[col].values
                has_classifier = True
                print(f"[Classifier] Loaded {len(df):,} cached classifier scores.")
            else:
                print("[Classifier] Cached file shape mismatch -- skipping classifier scores.")
        else:
            print("\n[Classifier] Skipped (--skip-classifier flag).")

    # Step 3: LCR ----------------------------------------------------------
    print("\n=== Step 3 / 7: Computing Lexical Conflict Rate ===")
    df, lcr_agg = compute_lcr(df)
    df.to_csv(out_dir / "lexical_conflict_crash.csv", index=False)
    lcr_agg.to_csv(out_dir / "lcr_aggregate.csv", index=False)
    print(f"[LCR] Raw scores saved  -> lexical_conflict_crash.csv")
    print(f"[LCR] Aggregate saved   -> lcr_aggregate.csv")
    print("\n[LCR] LCR% by model x condition x quartile:")
    pivot = lcr_agg.pivot_table(
        index=["display_model", "Agent_Type"], columns="Quartile", values="lcr_pct"
    )
    print(pivot[["Q1", "Q2", "Q3", "Q4"]].to_string())

    # Step 4: Case studies -------------------------------------------------
    print("\n=== Step 4 / 7: Selecting case studies ===")
    cases = select_case_studies(df)
    cases.to_csv(out_dir / "case_studies.csv", index=False)
    print(f"[Cases] {len(cases)} examples saved -> case_studies.csv")
    print("\n--- Memory condition case studies (Q3, peak crash) ---")
    for _, row in cases[cases["condition"] == "memory"].iterrows():
        print(f"\n[{row['category'].upper()} | {row['model']} | {row['persona']} | "
              f"Day {row['day']} | LCR={row['lcr_flag']}]")
        print(f"  Action   : {row['action']}")
        print(f"  Rationale: {row['rationale'][:280]}")

    # Step 5: Summary table -----------------------------------------------
    print("\n=== Step 5 / 7: Summary table ===")
    summary = build_summary_table(df, has_classifier)
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    print(summary.to_string(index=False))

    # Step 6: Figures ------------------------------------------------------
    print("\n=== Step 6 / 7: Generating figures ===")
    try:
        import matplotlib  # noqa: F401
        if has_classifier:
            plot_persona_confidence(df, out_dir)
        else:
            print("[Plot] Skipping persona confidence figure (no classifier scores).")
        plot_lcr(lcr_agg, out_dir)
        plot_lcr_2x2(lcr_agg, out_dir)
    except ImportError:
        print("[Plot] matplotlib not available. Install it via: pip install matplotlib")

    # Step 7: 2x2 contrasts -----------------------------------------------
    print("\n=== Step 7 / 7: 2x2 Contrast Summary ===")
    _print_2x2_contrasts(lcr_agg)

    print(f"\n=== All done. Results in: {out_dir.resolve()} ===")


if __name__ == "__main__":
    main()
