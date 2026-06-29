#!/usr/bin/env python3
"""
Persona-Classifier-Based Drift Metric
======================================

Trains a small BERT-family classifier on the rationale text emitted by
LLM agents during their first few simulated days (when the persona is
freshest), then measures *linguistic* drift by applying the classifier
to rationales from later days and tracking how confident it remains
that each rationale belongs to the intended persona.

This is the metric that turns FinPersona-Bench from a behavioral-only
evaluation into one with an independent NLP-grade drift signal —
addressing the "MSD is just behavioral, not linguistic" critique.

Pipeline
--------
  1. Load per-step CSVs from --results-dir (excludes master_summary
     + t_calibration).
  2. Filter to rationales from days 1..EARLY_DAYS (default 5).
  3. Train a 3-class DistilBERT (ENTJ / ISFJ / INTJ) with stratified
     train/val split, saved to --output-dir.
  4. Score every rationale (all days, all conditions); write
     drift_scores.csv with per-day classifier confidence in the
     intended persona.
  5. (Optional) `plot` subcommand generates decay curves.

Usage
-----
    # First-time training (run after some results have accumulated)
    python analysis/persona_classifier.py train \
        --results-dir results \
        --output-dir analysis/outputs/persona_classifier

    # Score all rationales using the trained classifier
    python analysis/persona_classifier.py score \
        --results-dir results \
        --output-dir analysis/outputs/persona_classifier

    # Plot decay curves once scores exist
    python analysis/persona_classifier.py plot \
        --output-dir analysis/outputs/persona_classifier
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Heavy deps imported lazily inside functions so plotting / scoring don't
# require torch+transformers to be present unless training.
DEFAULT_HF_MODEL = "distilbert-base-uncased"
PERSONAS         = ["ENTJ", "ISFJ", "INTJ"]
PERSONA_TO_ID    = {p: i for i, p in enumerate(PERSONAS)}
ID_TO_PERSONA    = {i: p for p, i in PERSONA_TO_ID.items()}

MAX_LEN          = 256          # rationales are 2-3 sentences; 256 tokens is plenty
DEFAULT_BATCH    = 16
DEFAULT_LR       = 2e-5
DEFAULT_EPOCHS   = 5
DEFAULT_EARLY    = 5            # use day 1..5 as "fresh persona" training data
DEFAULT_SEED     = 42


# ──────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────

def load_per_step_csvs(results_dir: Path,
                       exclude_substrings: List[str] = None) -> pd.DataFrame:
    """Read every per-step CSV under results_dir, concatenate with derived Day column."""
    if exclude_substrings is None:
        exclude_substrings = ["master_summary", "t_calibration", "pilot"]

    csv_files = [
        f for f in results_dir.rglob("*.csv")
        if not any(s in str(f) for s in exclude_substrings)
    ]
    if not csv_files:
        raise FileNotFoundError(
            f"No per-step CSVs found under {results_dir}. "
            f"Make sure run_experiments.py has produced some output first."
        )

    required = {"Date", "MBTI", "Rationale", "Agent_Type", "Model", "Scenario", "Seed"}
    dfs = []
    skipped = 0
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception:
            skipped += 1
            continue
        if not required.issubset(df.columns):
            skipped += 1
            continue
        # Derive integer Day from "Day-N" string
        df["Day"] = pd.to_numeric(
            df["Date"].astype(str).str.extract(r"Day-(\d+)", expand=False),
            errors="coerce",
        ).astype("Int64")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("All per-step CSVs were filtered out; nothing to load.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[Data] Loaded {len(csv_files) - skipped} CSVs, {len(combined):,} rows "
          f"(skipped {skipped}).")
    return combined


def filter_training_data(df: pd.DataFrame, early_days: int) -> pd.DataFrame:
    """Keep rationales from days 1..early_days, drop blanks, restrict to known personas."""
    df = df[df["MBTI"].isin(PERSONAS)].copy()
    df = df[df["Day"].between(1, early_days, inclusive="both")]
    df = df[df["Rationale"].notna()]
    df = df[df["Rationale"].astype(str).str.strip().str.len() >= 5]
    df["Rationale"] = df["Rationale"].astype(str)
    return df.reset_index(drop=True)


def balance_per_persona(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Cap rows per persona to keep training balanced + cheap."""
    rng = np.random.default_rng(seed)
    parts = []
    for p in PERSONAS:
        sub = df[df["MBTI"] == p]
        if len(sub) > max_per_class:
            idx = rng.choice(len(sub), max_per_class, replace=False)
            sub = sub.iloc[idx]
        parts.append(sub)
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
#  TRAINING
# ──────────────────────────────────────────────────────────────────────────

def _stratified_split(df: pd.DataFrame, val_frac: float, seed: int):
    """Stratified train/val split by MBTI."""
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for p in PERSONAS:
        sub = df[df["MBTI"] == p].sample(frac=1, random_state=seed).reset_index(drop=True)
        n_val = int(round(len(sub) * val_frac))
        val_parts.append(sub.iloc[:n_val])
        train_parts.append(sub.iloc[n_val:])
    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = pd.concat(val_parts,   ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def cmd_train(args):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix
    )
    from tqdm.auto import tqdm

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    print(f"[Train] HF base model: {args.hf_model}")

    # 1. Load + filter data
    raw = load_per_step_csvs(Path(args.results_dir))
    print(f"[Train] Total per-step rows available: {len(raw):,}")
    train_pool = filter_training_data(raw, args.early_days)
    print(f"[Train] Day 1..{args.early_days} rationales: {len(train_pool):,}")
    print(train_pool["MBTI"].value_counts().to_string())

    if args.max_samples_per_class:
        train_pool = balance_per_persona(
            train_pool, args.max_samples_per_class, seed=args.seed
        )
        print(f"[Train] After balancing to ≤{args.max_samples_per_class}/class: {len(train_pool):,}")

    train_df, val_df = _stratified_split(train_pool, val_frac=args.val_frac, seed=args.seed)
    print(f"[Train] Train: {len(train_df):,}   Val: {len(val_df):,}")

    # 2. Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.hf_model, num_labels=len(PERSONAS),
        id2label=ID_TO_PERSONA, label2id=PERSONA_TO_ID,
    ).to(device)

    # 3. Datasets
    class _RDataset(Dataset):
        def __init__(self, df):
            self.texts  = df["Rationale"].tolist()
            self.labels = df["MBTI"].map(PERSONA_TO_ID).astype(int).tolist()
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, i):
            enc = tokenizer(self.texts[i], truncation=True, max_length=MAX_LEN,
                            padding="max_length", return_tensors="pt")
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(self.labels[i], dtype=torch.long),
            }

    train_loader = DataLoader(_RDataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(_RDataset(val_df),   batch_size=args.batch_size, shuffle=False)

    # 4. Training loop
    optim = AdamW(model.parameters(), lr=args.lr)
    history = []

    def evaluate():
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
                probs  = torch.softmax(logits, dim=-1)
                preds  = probs.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
        return all_labels, all_preds, all_probs

    print("[Train] Starting fine-tune…")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optim.step()
            running += out.loss.item() * len(batch["labels"])
            n += len(batch["labels"])
            pbar.set_postfix(loss=f"{running/n:.4f}")

        # Validate
        labels, preds, _ = evaluate()
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average="macro")
        print(f"  epoch {epoch+1}: train_loss={running/n:.4f}  val_acc={acc:.4f}  val_macroF1={f1:.4f}")
        history.append({"epoch": epoch+1, "train_loss": running/n, "val_acc": acc, "val_macroF1": f1})

    # 5. Final evaluation
    labels, preds, _ = evaluate()
    final_acc = accuracy_score(labels, preds)
    final_f1  = f1_score(labels, preds, average="macro")
    report    = classification_report(labels, preds, target_names=PERSONAS, digits=4)
    cm        = confusion_matrix(labels, preds).tolist()

    print("\n[Train] Final classification report:")
    print(report)

    # 6. Save
    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")

    metrics = {
        "hf_model":     args.hf_model,
        "early_days":   args.early_days,
        "n_train":      len(train_df),
        "n_val":        len(val_df),
        "val_accuracy": final_acc,
        "val_macroF1":  final_f1,
        "history":      history,
        "personas":     PERSONAS,
        "report":       report,
        "confusion_matrix": cm,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[Train] Saved → {out_dir}/")
    print(f"        model/   metrics.json")
    if final_acc < 0.85:
        print(f"  WARNING: val_acc={final_acc:.3f} is below 0.85 — classifier may be too weak "
              f"to reliably detect drift. Try more samples or a larger HF model.")


# ──────────────────────────────────────────────────────────────────────────
#  SCORING
# ──────────────────────────────────────────────────────────────────────────

def cmd_score(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm.auto import tqdm

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "model"
    if not model_dir.exists():
        print(f"ERROR: no trained model at {model_dir}. Run `train` first.")
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    raw = load_per_step_csvs(Path(args.results_dir))
    raw = raw[raw["MBTI"].isin(PERSONAS)].copy()
    raw = raw[raw["Rationale"].notna() & (raw["Rationale"].astype(str).str.strip().str.len() >= 5)]
    raw["Rationale"] = raw["Rationale"].astype(str)
    print(f"[Score] Scoring {len(raw):,} rationales…")

    texts = raw["Rationale"].tolist()
    probs_all = np.zeros((len(texts), len(PERSONAS)), dtype=np.float32)
    batch = args.batch_size

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch), desc="scoring"):
            chunk = texts[start:start + batch]
            enc = tokenizer(chunk, truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_all[start:start + batch] = probs

    # Per-row: probability of the *intended* persona (the MBTI column)
    intended_idx = raw["MBTI"].map(PERSONA_TO_ID).astype(int).to_numpy()
    p_intended   = probs_all[np.arange(len(raw)), intended_idx]
    p_argmax_id  = probs_all.argmax(axis=1)
    p_argmax     = np.array([ID_TO_PERSONA[int(i)] for i in p_argmax_id])
    correct      = (p_argmax_id == intended_idx).astype(int)

    out = pd.DataFrame({
        "Model":            raw["Model"].values,
        "Agent_Type":       raw["Agent_Type"].values,
        "Scenario":         raw["Scenario"].values,
        "Seed":             raw["Seed"].values,
        "MBTI":             raw["MBTI"].values,
        "Day":              raw["Day"].values,
        "p_intended":       p_intended,
        "predicted_persona": p_argmax,
        "correct":          correct,
    })

    # Also add per-persona probability columns for richer downstream plots
    for i, p in enumerate(PERSONAS):
        out[f"p_{p}"] = probs_all[:, i]

    score_path = out_dir / "drift_scores.csv"
    out.to_csv(score_path, index=False)
    print(f"[Score] Saved → {score_path}  ({len(out):,} rows)")

    # Quick summary
    print("\n[Score] Summary — overall classifier-correct rate per (Agent_Type, Day-bucket):")
    out["DayBucket"] = pd.cut(out["Day"].astype(float),
                              bins=[0, 25, 50, 75, 100, 200, 400, 1000],
                              labels=["1-25", "26-50", "51-75", "76-100",
                                      "101-200", "201-400", "401+"],
                              include_lowest=True)
    pivot = out.pivot_table(index="DayBucket", columns="Agent_Type",
                            values="correct", aggfunc="mean", observed=True)
    print(pivot.round(3).to_string())


# ──────────────────────────────────────────────────────────────────────────
#  PLOT (decay curves)
# ──────────────────────────────────────────────────────────────────────────

def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.output_dir)
    scores = pd.read_csv(out_dir / "drift_scores.csv")

    # Smooth via rolling mean across days within each (Model, MBTI, Agent_Type)
    # group; aggregate across seeds.
    metric = "p_intended"   # confidence in the intended persona
    window = max(5, int(scores["Day"].max() / 40))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, persona in zip(axes, PERSONAS):
        sub = scores[scores["MBTI"] == persona]
        if sub.empty:
            ax.set_title(f"{persona} (no data)")
            continue

        for agent_type, color in [("static", "#E84855"), ("memory", "#1B998B")]:
            cond = sub[sub["Agent_Type"] == agent_type]
            if cond.empty:
                continue
            curve = (cond.groupby("Day")[metric].mean()
                          .rolling(window, min_periods=1).mean())
            ax.plot(curve.index, curve.values, label=agent_type,
                    color=color, linewidth=2.2)

            # 95% CI band from seeds
            grp = cond.groupby("Day")[metric].agg(["mean", "std", "count"])
            grp["se"]  = grp["std"] / np.sqrt(grp["count"].clip(lower=1))
            grp["lo"]  = (grp["mean"] - 1.96 * grp["se"]).rolling(window, min_periods=1).mean()
            grp["hi"]  = (grp["mean"] + 1.96 * grp["se"]).rolling(window, min_periods=1).mean()
            ax.fill_between(grp.index, grp["lo"], grp["hi"], color=color, alpha=0.15)

        ax.set_title(f"Persona: {persona}", fontweight="bold")
        ax.set_xlabel("Simulation day")
        ax.set_ylabel("P(intended persona)")
        ax.set_ylim(0, 1.05)
        ax.axhline(1 / len(PERSONAS), color="gray", linestyle=":", linewidth=1, label="random baseline")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9, frameon=True)

    fig.suptitle("Linguistic Persona Drift — classifier confidence vs. simulation day",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = out_dir / "decay_curves.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {fig_path}")


# ──────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # train ──────────────────────────
    pt = sub.add_parser("train", help="Fine-tune a persona classifier on early-day rationales")
    pt.add_argument("--results-dir", default="results")
    pt.add_argument("--output-dir",  default="analysis/outputs/persona_classifier")
    pt.add_argument("--hf-model",    default=DEFAULT_HF_MODEL)
    pt.add_argument("--early-days",  type=int, default=DEFAULT_EARLY,
                    help=f"Train on day 1..N rationales (default {DEFAULT_EARLY})")
    pt.add_argument("--epochs",      type=int, default=DEFAULT_EPOCHS)
    pt.add_argument("--batch-size",  type=int, default=DEFAULT_BATCH)
    pt.add_argument("--lr",          type=float, default=DEFAULT_LR)
    pt.add_argument("--val-frac",    type=float, default=0.15)
    pt.add_argument("--max-samples-per-class", type=int, default=2000,
                    help="Cap training samples per persona (default 2000)")
    pt.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    pt.set_defaults(func=cmd_train)

    # score ──────────────────────────
    ps = sub.add_parser("score", help="Apply trained classifier to all rationales")
    ps.add_argument("--results-dir", default="results")
    ps.add_argument("--output-dir",  default="analysis/outputs/persona_classifier")
    ps.add_argument("--batch-size",  type=int, default=64)
    ps.set_defaults(func=cmd_score)

    # plot ───────────────────────────
    pp = sub.add_parser("plot", help="Generate decay curves from drift_scores.csv")
    pp.add_argument("--output-dir",  default="analysis/outputs/persona_classifier")
    pp.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
