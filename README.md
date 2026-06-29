# <img width="80" alt="FinPersona Logo" src="https://github.com/user-attachments/assets/5e09882c-f303-4fff-a112-438520448771" /> FinPersona-Bench: A Benchmark for Longitudinal Psychometric Stability of Autonomous Financial Agents

FinPersona-Bench is a simulation benchmark for measuring **Mandate Salience Decay (MSD)** — the tendency of LLM-based financial agents to gradually drift from their behavioral mandates as market context accumulates over long horizons. A synthetic market decouples observable price from hidden fundamental value, enabling falsifiable evaluation of 18 frontier and open-source LLMs across three market regimes and three behavioral profiles.

---

## Architecture

### 1. Market Environment (`envs/`)

Generates synthetic financial data with three scenario types:
- **`flat`** — Regime-based GARCH-like volatility clustering; low signal-to-noise baseline.
- **`bull_trap`** — Multi-phase bubble: legitimate rise → mania → blow-off top.
- **`crash`** — Panic selling: fundamental deterioration → oversold panic → stabilization.

Each scenario exposes per-step observables (price, SMA20/SMA60, RSI14, P/E, implied volatility, volume ratio, news sentiment) and a hidden `Fundamental_Value` used for rationality scoring.

### 2. Agent Layer (`agent/`)

Four agent types built on a shared `BaseAgent` interface:

| Agent | Description |
|-------|-------------|
| `StaticAgent` | Baseline. Persona injected once at init; never re-injected. |
| `ActiveMemoryAgent` | Extends Static with **Periodic Mandate Retrieval**: core persona mandate re-injected at every decision step. |
| `PlaceboAgent` | Re-injects semantically irrelevant boilerplate each step — matches the token-overhead of memory without reinstating the mandate. Used as a three-arm control. |
| `InjectionFreqAgent` | Parameterized injection schedule (k ∈ {1, 5, 25, 100, ∞}) with a per-step `Mandate_Injected` audit column. |

Structured output (action, quantity, rationale) is enforced via **Pydantic** schemas. 

Personality profiles are defined in `agent/personas/mbti_profiles.json` and cover all 16 MBTI types plus `EXPERT` and `NONE` baselines, and the four OCEAN archetypes (`O1_conservative`, `O2_aggressive`, `O3_conservative`, `O3_aggressive`).

### 3. Simulation & Execution (`simulation/`)

- **`runner.py`** — Main orchestrator. Connects environment to agent, manages the trading loop (100–800 days), and logs portfolio state + market observables daily to CSV.
- **`portfolio_tracker.py`** — Accounting module tracking cash, shares, mark-to-market value, and a full transaction log.

### 4. Benchmarking (`run_experiments.py`)

Full experimental matrix with concurrent execution (up to 10 parallel workers):
- **Personas:** 3 primary (ENTJ, ISFJ, INTJ); expandable to all 16 MBTI types
- **Scenarios:** `flat`, `bull_trap`, `crash` (+ crash sensitivity at discounts 0.85/0.92/0.95)
- **Agent types:** `static`, `memory`
- **Seeds:** 5 (42, 123, 456, 789, 999)
- **Checkpointing:** Completed runs are tracked in `checkpoint.txt` so experiments are fully resumable.

### 5. Analysis (`analysis_may/`)

- **`numerical_analysis.py`** — Computes financial metrics (Return %, Max Drawdown), rationality score, stereotype metrics (trade count), drift (MAS Deviation from `cideal`), and bubble participation (Avg Buy P/E). Produces per-metric, per-model, and per-persona breakdowns with Wilcoxon signed-rank significance tests.
- **`analysis_may/outputs/`** contains all generated figures and tables:
  - `figures/hero_figure.{png,pdf}` — Oral-grade 3-panel linguistic decay curve with 95% CI bands and effect-size annotations.
  - `effect_sizes/effect_size_table.csv` + `forest_plot.png` — Paired Cliff's δ and Hedges' g for all scenario × metric cells.
  - `injection_freq/injection_freq_curve.{csv,png,pdf}` — Adherence vs. injection frequency Pareto curve.
  - `persona_classifier/decay_curves.png` + `drift_scores.csv` — DistilBERT classifier adherence trajectories over the simulation horizon.
  - `rationale_linguistic/lcr_aggregate.csv` — Lexical Conflict Rate analysis on crash scenario.

---

## Experiments

### Main Panel — Open-Weight & API Models (`results_may/`)

Extends the original 3-model API-only panel to a **9-model panel** spanning 4B–frontier parameters across two access classes:

| Model | Family | Access | Params |
|-------|--------|--------|--------|
| `google/gemma-3-4b-it` | Gemma 3 | open-weight (vLLM) | 4 B |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen 2.5 | open-weight (vLLM) | 7 B |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 | open-weight (vLLM) | 8 B |
| `google/gemma-2-9b-it` | Gemma 2 | open-weight (vLLM) | 9 B |
| `Qwen/Qwen2.5-14B-Instruct` | Qwen 2.5 | open-weight (vLLM) | 14 B |
| `google/gemma-3-27b-it` | Gemma 3 | open-weight (vLLM) | 27 B |
| `gemini-2.5-flash` | Gemini 2.5 | API (Google) | — |
| `gemini-3-flash-preview` | Gemini 3 | API (Google) | — |
| `claude-sonnet-4-6` | Claude Sonnet 4 | API (Anthropic) | — |

**Design:** 9 models × 3 personas × 3 scenarios × 2 agent types × 5 seeds = ~810 simulations, T=200 days, temperature=0.2. The 6 open-weight models are fully reproducible without API access.

### OCEAN Framework-Independence Replication (`results_ocean/`, `analysis_ocean/`)

Tests whether the MBTI drift/recovery pattern generalizes to **OCEAN (Big Five)** personality scaffolding. Two behavioral archetypes plus a numerical-only ablation (O3) are evaluated across three API models (`claude-sonnet-4-6`, `gpt-4o-mini`, `gemini-2.5-flash`) on all three scenarios.

| Persona | Description | `cideal` |
|---------|-------------|---------|
| `O1_conservative` | High-agreeableness, risk-averse | 1.0 |
| `O2_aggressive` | High-extraversion, risk-seeking | 0.2 |
| `O3_conservative` | Numerical-only conservative (ablation) | 1.0 |
| `O3_aggressive` | Numerical-only aggressive (ablation) | 0.2 |

### Placebo Re-injection Control (`placebo_reinjection_control/`)

A three-arm causal study isolating *mandate salience* as the active ingredient of memory re-injection:

| Arm | What is re-injected | Purpose |
|-----|---------------------|---------|
| **Static** | Nothing (mandate at init only) | Baseline drift condition |
| **Placebo** | Semantically irrelevant boilerplate each step | Controls for token attention from any re-injection |
| **Memory** | Persona mandate each step | The intervention |

Run on `claude-sonnet-4-6`, flat scenario, all three personas across 5 seeds (15 new simulations). If placebo ≈ static and memory > placebo, the benefit is specific to mandate content rather than mere text volume.

### Injection Frequency Ablation (`results_may/injection_freq/`)

Sweeps mandate-refresh cadence k ∈ {1, 5, 25, 100, ∞} to characterize the Pareto trade-off between token overhead and persona adherence. k=∞ is the static baseline; k=1 is the memory agent. Run on `Qwen/Qwen2.5-7B-Instruct`, flat scenario, all three personas, seeds 42/123/456 (45 simulations total, T=200). Each per-step CSV records a `Mandate_Injected` column as an audit trail.

### Rationality & Linguistic Analysis (`rationale_linguistic_analysis/`)

Provides linguistic evidence for signal interference in mini-model crash reversals using existing rationale strings — no new simulations required. Two complementary analyses:

1. **Persona Classifier Confidence Trajectories** — DistilBERT classifier applied to crash-scenario rationales, segmented by quartile, for mini vs. flagship models under static vs. memory conditions.
2. **Lexical Conflict Rate (LCR)** — counts rationales that co-express mandate-aligned language *and* market-reactive stress language simultaneously.

Models studied: `gpt-4o-mini`, `gpt-4.1-mini` (mini); `gpt-4o`, `claude-sonnet-4-6` (flagship). Scenario: crash/discount0.92.

### Long-Horizon T=800 (`results_may/long_horizon/`)

Extends the simulation horizon to T=800 to connect with long-context NLP evaluation standards. Since the agent has no knowledge of its stopping point, the first 100/200/400 days of a T=800 trajectory are statistically equivalent to standalone runs at those horizons — one dataset yields decay curves at every shorter horizon. Run on `google/gemma-2-9b-it` and `google/gemma-3-27b-it`, flat scenario, seeds 42/123/456.

---

## Setup

**Prerequisites:** Python 3.9+ and API keys for the model providers you intend to use.

1. Clone the repository:
```bash
git clone https://github.com/ayeshag7/FinPersona.git
cd FinPersona
```

2. Install dependencies:
```bash
pip install -e .[dev]
```

3. Create a `.env` file in the root directory with your API keys:
```bash
GOOGLE_API_KEY=your_google_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```
Only include keys for the providers you plan to use. For open-weight models, a vLLM server must be running locally.

## Usage

**Single backtest run** (ENTJ persona, 50 days):
```bash
python run_backtest.py
```

**Full benchmark suite** (all models × personas × scenarios × seeds):
```bash
python run_experiments.py
```

**Human annotation app:**
```bash
streamlit run annotation_app/annotation_app.py
```
