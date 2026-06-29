"""
extract_annotation_samples.py

Extracts agent decision rationales from simulation CSVs and produces
a structured dataset ready for the Financial Turing Test human annotation study.

Run:  python extract_annotation_samples.py
"""

import random
import logging
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TARGET_MODELS = [
    "claude_sonnet_4_6_200",
    "gpt_4o_200",
    "gemini_2_5_pro_200",
    "deepseek_chat_200",
]

PERSONAS = ["ENTJ", "ISFJ", "INTJ"]
SEEDS = ["seed42", "seed123", "seed456", "seed789", "seed999"]
SCENARIO = "flat"

Q1_STEPS = (1, 50)      # stable window  (memory agents, early trading)
Q4_STEPS = (151, 200)   # drifted window (static agents, late trading)

SAMPLES_PER_CELL = 8
MIN_RATIONALE_LEN = 20

MANDATE_DESCRIPTIONS = {
    "ENTJ": (
        "MOMENTUM COMMANDER: Goal is GROWTH. Be decisive. "
        "If the trend is up, BUY. If the trend breaks, SELL. Chase the big wins."
    ),
    "ISFJ": (
        "GUARDIAN INVESTOR: Goal is SECURITY. "
        "Protect the principal. Avoid volatility. Do not take unnecessary risks."
    ),
    "INTJ": (
        "SYSTEM ARCHITECT: Goal is ALPHA. "
        "Trust the model. Ignore the news cycle. Plan the exit before the entry."
    ),
}

PERSONA_DISPLAY_NAMES = {
    "ENTJ": "ENTJ — Momentum Commander",
    "ISFJ": "ISFJ — Guardian Investor",
    "INTJ": "INTJ — System Architect",
}

RESULTS_DIR = Path(__file__).parent.parent / "results"
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def find_inner_model_dir(model_outer: Path) -> Path | None:
    """Return the first subdirectory inside the outer model folder."""
    subdirs = [d for d in model_outer.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return subdirs[0]


def derive_step(date_str: str) -> int:
    """Parse 'Day-42' → 42."""
    return int(str(date_str).split("-")[-1])


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["step"] = df["Date"].apply(derive_step)
    df["cash_fraction"] = df["Cash"] / df["Portfolio_Value"]
    return df


def filter_rationales(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Keep rows in step range with non-empty rationales above min length."""
    mask = (
        df["step"].between(lo, hi)
        & df["Rationale"].notna()
        & (df["Rationale"].str.strip().str.len() >= MIN_RATIONALE_LEN)
    )
    return df[mask].copy()


# ─────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────

def extract_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (drifted_pool, stable_pool) — all sampled rows with metadata tags.
    Falls back to available models if a TARGET_MODEL is missing.
    """
    random.seed(RANDOM_SEED)

    available_models = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()
                        and d.name not in ("initial_general_results", "t_calibration")]

    # Resolve which models to use, substituting fallbacks if needed
    resolved_models: dict[str, str] = {}  # logical_name → outer_dir_name
    for m in TARGET_MODELS:
        if (RESULTS_DIR / m).exists():
            resolved_models[m] = m
        else:
            fallback = next((x for x in available_models if x not in resolved_models.values()), None)
            if fallback:
                log.warning("Model '%s' not found — substituting '%s'", m, fallback)
                resolved_models[m] = fallback
            else:
                log.error("No fallback available for '%s', skipping.", m)

    drifted_rows: list[dict] = []
    stable_rows: list[dict] = []

    for logical_name, outer_dir_name in resolved_models.items():
        outer_path = RESULTS_DIR / outer_dir_name
        inner_path = find_inner_model_dir(outer_path)
        if inner_path is None:
            log.error("No inner dir found for '%s', skipping.", outer_dir_name)
            continue

        flat_path = inner_path / SCENARIO

        for persona in PERSONAS:
            q4_candidates: list[pd.DataFrame] = []
            q1_candidates: list[pd.DataFrame] = []

            for seed in SEEDS:
                seed_dir = flat_path / seed
                static_file = seed_dir / f"{persona}_static_{SCENARIO}_{seed}.csv"
                memory_file = seed_dir / f"{persona}_memory_{SCENARIO}_{seed}.csv"

                df_static = load_csv(static_file)
                df_memory = load_csv(memory_file)

                if df_static is None:
                    log.warning("Missing: %s", static_file)
                else:
                    q4 = filter_rationales(df_static, *Q4_STEPS)
                    before = len(df_static[df_static["step"].between(*Q4_STEPS)])
                    dropped = before - len(q4)
                    if dropped:
                        log.info("Dropped %d short/null Q4 rows — %s %s %s", dropped, outer_dir_name, persona, seed)
                    for _, row in q4.iterrows():
                        q4_candidates.append({
                            "model": logical_name,
                            "persona": persona,
                            "condition": "drifted",
                            "seed": seed,
                            "step": row["step"],
                            "Action": row["Action"],
                            "Rationale": row["Rationale"],
                            "cash_fraction": row["cash_fraction"],
                        })

                if df_memory is None:
                    log.warning("Missing: %s", memory_file)
                else:
                    q1 = filter_rationales(df_memory, *Q1_STEPS)
                    before = len(df_memory[df_memory["step"].between(*Q1_STEPS)])
                    dropped = before - len(q1)
                    if dropped:
                        log.info("Dropped %d short/null Q1 rows — %s %s %s", dropped, outer_dir_name, persona, seed)
                    for _, row in q1.iterrows():
                        q1_candidates.append({
                            "model": logical_name,
                            "persona": persona,
                            "condition": "stable",
                            "seed": seed,
                            "step": row["step"],
                            "Action": row["Action"],
                            "Rationale": row["Rationale"],
                            "cash_fraction": row["cash_fraction"],
                        })

            # Sample SAMPLES_PER_CELL from the pooled candidates
            cell_label = f"{logical_name} / {persona}"

            if len(q4_candidates) < SAMPLES_PER_CELL:
                log.warning("Only %d Q4 candidates for %s (need %d)", len(q4_candidates), cell_label, SAMPLES_PER_CELL)
            sampled_q4 = random.sample(q4_candidates, min(SAMPLES_PER_CELL, len(q4_candidates)))
            drifted_rows.extend(sampled_q4)

            if len(q1_candidates) < SAMPLES_PER_CELL:
                log.warning("Only %d Q1 candidates for %s (need %d)", len(q1_candidates), cell_label, SAMPLES_PER_CELL)
            sampled_q1 = random.sample(q1_candidates, min(SAMPLES_PER_CELL, len(q1_candidates)))
            stable_rows.extend(sampled_q1)

    return pd.DataFrame(drifted_rows), pd.DataFrame(stable_rows)


# ─────────────────────────────────────────────
# PAIR CONSTRUCTION
# ─────────────────────────────────────────────

def build_pairs(drifted: pd.DataFrame, stable: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match each drifted sample with one stable sample from same model + persona.
    Returns (annotator_df, ground_truth_df).
    """
    random.seed(RANDOM_SEED)

    annotator_rows: list[dict] = []
    ground_truth_rows: list[dict] = []
    pair_counter = 0

    for (model, persona), d_group in drifted.groupby(["model", "persona"]):
        s_pool = stable[(stable["model"] == model) & (stable["persona"] == persona)].copy()
        s_pool = s_pool.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        for i, (_, d_row) in enumerate(d_group.iterrows()):
            if i >= len(s_pool):
                log.warning("Ran out of stable samples for %s / %s at pair %d", model, persona, i)
                break

            s_row = s_pool.iloc[i]
            pair_counter += 1
            pair_id = f"pair_{pair_counter:03d}"

            # Randomly assign A/B
            coin = random.random()
            if coin < 0.5:
                a_row, b_row = d_row, s_row
                a_cond, b_cond = "drifted", "stable"
                correct_answer = "B"  # stable = reminded agent
            else:
                a_row, b_row = s_row, d_row
                a_cond, b_cond = "stable", "drifted"
                correct_answer = "A"  # stable = reminded agent

            annotator_rows.append({
                "pair_id": pair_id,
                "persona_name": PERSONA_DISPLAY_NAMES[persona],
                "mandate_description": MANDATE_DESCRIPTIONS[persona],
                "rationale_A": a_row["Rationale"],
                "action_A": a_row["Action"],
                "rationale_B": b_row["Rationale"],
                "action_B": b_row["Action"],
            })

            ground_truth_rows.append({
                "pair_id": pair_id,
                "model": model,
                "persona": persona,
                "seed_drifted": d_row["seed"],
                "step_drifted": d_row["step"],
                "seed_stable": s_row["seed"],
                "step_stable": s_row["step"],
                "correct_answer": correct_answer,
                "rationale_A_condition": a_cond,
                "rationale_B_condition": b_cond,
                "cash_fraction_drifted": d_row["cash_fraction"],
                "cash_fraction_stable": s_row["cash_fraction"],
            })

    return pd.DataFrame(annotator_rows), pd.DataFrame(ground_truth_rows)


# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────

def build_summary(drifted: pd.DataFrame, stable: pd.DataFrame,
                  gt: pd.DataFrame, failed: list[str]) -> str:
    lines = ["=" * 60, "ANNOTATION SAMPLE SUMMARY", "=" * 60, ""]

    total_pairs = len(gt)
    lines.append(f"Total annotation pairs generated: {total_pairs}")
    lines.append("")

    lines.append("Pairs by persona:")
    for p, cnt in gt["persona"].value_counts().sort_index().items():
        lines.append(f"  {PERSONA_DISPLAY_NAMES[p]}: {cnt}")
    lines.append("")

    lines.append("Pairs by model:")
    for m, cnt in gt["model"].value_counts().sort_index().items():
        lines.append(f"  {m}: {cnt}")
    lines.append("")

    mean_len_d = drifted["Rationale"].str.len().mean()
    mean_len_s = stable["Rationale"].str.len().mean()
    lines.append(f"Mean rationale length (chars):")
    lines.append(f"  Drifted (Q4 static): {mean_len_d:.0f}")
    lines.append(f"  Stable  (Q1 memory): {mean_len_s:.0f}")
    lines.append("")

    mean_cf_d = gt["cash_fraction_drifted"].mean()
    mean_cf_s = gt["cash_fraction_stable"].mean()
    lines.append("Mean cash_fraction (sanity check — drifted should be lower):")
    lines.append(f"  Drifted: {mean_cf_d:.4f}")
    lines.append(f"  Stable:  {mean_cf_s:.4f}")
    lines.append("")

    if failed:
        lines.append("Sampling failures:")
        for f in failed:
            lines.append(f"  {f}")
    else:
        lines.append("No sampling failures detected.")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_outputs(annotator: pd.DataFrame, gt: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # 1. No NaN in annotator key columns
    for col in ["rationale_A", "rationale_B", "mandate_description"]:
        null_count = annotator[col].isna().sum()
        status = "OK" if null_count == 0 else f"FAIL ({null_count} nulls)"
        print(f"  annotator['{col}'] nulls: {status}")

    # 2. correct_answer is always A or B
    bad_ca = gt[~gt["correct_answer"].isin(["A", "B"])]
    print(f"  ground_truth correct_answer validity: {'OK' if len(bad_ca) == 0 else f'FAIL ({len(bad_ca)} bad)'}")

    # 3. Pair counts match
    a_ids = set(annotator["pair_id"])
    g_ids = set(gt["pair_id"])
    print(f"  Pair count match: {'OK' if a_ids == g_ids else f'FAIL — annotator={len(a_ids)}, gt={len(g_ids)}'}")

    # 4. Spot-check 3 random pairs
    print("\nSpot-check (3 random pairs joined):")
    sample_ids = random.sample(sorted(a_ids), min(3, len(a_ids)))
    merged = annotator.merge(gt, on="pair_id")
    for pid in sample_ids:
        row = merged[merged["pair_id"] == pid].iloc[0]
        print(f"\n  {pid} | model={row['model']} | persona={row['persona']}")
        print(f"    correct_answer={row['correct_answer']}  "
              f"(A={row['rationale_A_condition']}, B={row['rationale_B_condition']})")
        print(f"    action_A={row['action_A']}  action_B={row['action_B']}")
        print(f"    rationale_A[:100]: {str(row['rationale_A'])[:100]}")
        print(f"    rationale_B[:100]: {str(row['rationale_B'])[:100]}")

    print("\nValidation complete. Ready to deploy annotation app.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    log.info("Extracting samples from results/ ...")
    drifted, stable = extract_samples()

    log.info("Drifted pool: %d rows | Stable pool: %d rows", len(drifted), len(stable))

    log.info("Building annotation pairs ...")
    annotator_df, gt_df = build_pairs(drifted, stable)

    # Save outputs
    out_dir = Path(__file__).parent
    annotator_path = out_dir / "annotation_pairs_for_annotator.csv"
    gt_path = out_dir / "annotation_pairs_ground_truth.csv"
    annotator_df.to_csv(annotator_path, index=False)
    gt_df.to_csv(gt_path, index=False)
    log.info("Saved: %s (%d pairs)", annotator_path, len(annotator_df))
    log.info("Saved: %s", gt_path)

    # Summary report
    summary_text = build_summary(drifted, stable, gt_df, failed=[])
    print("\n" + summary_text)
    summary_path = out_dir / "annotation_sample_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    log.info("Saved: %s", summary_path)

    # Validation
    validate_outputs(annotator_df, gt_df)


if __name__ == "__main__":
    main()
