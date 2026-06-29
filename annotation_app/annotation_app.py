"""
annotation_app.py — FinPersona Financial Turing Test annotation interface

Launch:  streamlit run annotation_app.py
"""

import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ANNOTATOR_CSV = Path("annotation_pairs_for_annotator.csv")
RESULTS_DIR = Path("annotation_results")

PERSONA_ICONS = {
    "ENTJ": "",
    "ISFJ": "",
    "INTJ": "",
}

ACTION_ICONS = {
    "BUY": "📈",
    "SELL": "📉",
    "HOLD": "⏸️",
}

st.set_page_config(
    page_title="FinPersona — Financial Turing Test",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — works in both light and dark mode
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Rationale cards: no hardcoded bg so they inherit theme bg ── */
    .rcard {
        border: 1.5px solid rgba(150, 150, 150, 0.35);
        border-radius: 10px;
        padding: 18px 20px;
        height: 280px;
        overflow-y: auto;
        font-size: 0.91rem;
        line-height: 1.7;
        word-break: break-word;
    }

    /* ── Action badge ── */
    .action-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        border: 1.5px solid rgba(150,150,150,0.4);
        margin-bottom: 10px;
    }

    /* ── Mandate box ── */
    .mandate-box {
        border-left: 4px solid #6c8ebf;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        font-style: italic;
        margin-bottom: 8px;
        border-top: 1px solid rgba(150,150,150,0.2);
        border-right: 1px solid rgba(150,150,150,0.2);
        border-bottom: 1px solid rgba(150,150,150,0.2);
    }

    /* ── Card header ── */
    .card-header {
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 6px;
    }

    /* ── Landing page center ── */
    .landing-title {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }
    .landing-sub {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1rem;
    }

    /* ── Progress step dots ── */
    .step-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(108,142,191,0.18);
        border: 1px solid rgba(108,142,191,0.45);
        color: #6c8ebf;
        margin-bottom: 6px;
    }

    /* ── Question section ── */
    .q-label {
        font-weight: 700;
        font-size: 0.97rem;
        margin-bottom: 2px;
    }

    /* Reduce default top padding of main content area */
    .block-container {
        padding-top: 3.5rem !important;
    }

    /* Hide default streamlit branding in sidebar */
    [data-testid="stSidebarNav"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

@st.cache_data
def load_pairs() -> pd.DataFrame:
    return pd.read_csv(ANNOTATOR_CSV)


def save_response_incremental(pair_id: str, q1: str, q2: str, comments: str, elapsed: float) -> None:
    """Append a single response row to the annotator's running CSV."""
    annotator = st.session_state.annotator_name.replace(" ", "_")
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{annotator}_{st.session_state.session_id}.csv"

    row = pd.DataFrame([{
        "pair_id": pair_id,
        "annotator": st.session_state.annotator_name,
        "q1_answer": q1,
        "q2_answer": q2,
        "comments": comments,
        "time_taken_seconds": elapsed,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }])

    if out_path.exists():
        row.to_csv(out_path, mode="a", header=False, index=False)
    else:
        row.to_csv(out_path, index=False)


def persona_short(persona_name: str) -> str:
    """Extract ENTJ/ISFJ/INTJ from display name."""
    return persona_name.split("—")[0].strip().split()[-1]


def action_display(action: str) -> str:
    icon = ACTION_ICONS.get(str(action).upper(), "•")
    return f"{icon} {action}"


# ─────────────────────────────────────────────
# GUARD: CSV must exist
# ─────────────────────────────────────────────

if not ANNOTATOR_CSV.exists():
    st.error(
        "**Data file not found.**  \n"
        f"Expected `{ANNOTATOR_CSV}` — please run:  \n"
        "```\npython extract_annotation_samples.py\n```"
    )
    st.stop()

pairs_df = load_pairs()
TOTAL_PAIRS = len(pairs_df)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

for key, default in [
    ("annotator_name", ""),
    ("session_id", ""),
    ("study_started", False),
    ("pair_order", list(range(TOTAL_PAIRS))),
    ("current_idx", 0),
    ("responses", {}),
    ("pair_start_time", time.time()),
    ("scroll_to_top", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# SIDEBAR (shown only after study starts)
# ─────────────────────────────────────────────

if st.session_state.study_started:
    with st.sidebar:
        st.markdown("## 💹 FinPersona")
        st.caption("Financial Turing Test")
        st.divider()

        st.caption(
            "Benchmark testing whether AI trading agents maintain their investment personas. "
            "For each pair, judge which rationale better follows its mandate and which agent was just reminded of it."
        )
        st.divider()

        completed = len(st.session_state.responses)
        current_pair = min(st.session_state.current_idx + 1, TOTAL_PAIRS)
        pct = completed / TOTAL_PAIRS
        st.progress(pct)
        st.markdown(f"**Pair {current_pair} of {TOTAL_PAIRS}**")
        st.caption(f"{completed} completed")
        st.divider()

        st.markdown("**Annotator**")
        st.markdown(f"👤 {st.session_state.annotator_name}")
        st.divider()

        st.markdown(
            "**Instructions**\n\n"
            "1. Read both rationales carefully.\n"
            "2. Answer **both** questions.\n"
            "3. Press **Next** to continue.\n\n"
            "No time limit. Take your time."
        )

# ─────────────────────────────────────────────
# LANDING SCREEN
# ─────────────────────────────────────────────

if not st.session_state.study_started:
    # Hide sidebar on landing
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none}</style>",
        unsafe_allow_html=True,
    )

    _, centre, _ = st.columns([0.3, 3.4, 0.3])
    with centre:
        content_col, form_col = st.columns([1.4, 1], gap="large")

        with content_col:
            st.markdown(
                '<div class="landing-title">💹 FinPersona</div>'
                '<div style="font-size:1.35rem;font-weight:600;opacity:0.8;margin-bottom:0.4rem;">'
                "Financial Turing Test"
                "</div>"
                '<div class="landing-sub">Human Annotation Study</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"You will be shown **{TOTAL_PAIRS} pairs** of investment rationales generated by AI agents "
                "with distinct trading personas. Each pair shares the same market event but comes from agents "
                "with different investment mandates.\n\n"
                "For each pair, answer two questions:\n"
                "- **Mandate Consistency:** Which rationale better follows the agent's stated investment mandate?\n"
                "- **Mandate Reminder:** Which agent was shown its mandate again just before generating its rationale?"
            )

        with form_col:
            st.markdown("")
            with st.container(border=True):
                st.markdown("**Enter your name to begin**")
                st.text_input(
                    "Name",
                    placeholder="e.g. Jane Smith",
                    label_visibility="collapsed",
                    key="landing_name",
                )
                btn = st.button(
                    "Begin Study",
                    type="primary",
                    use_container_width=True,
                )

            st.caption("Responses are saved automatically after each pair to `annotation_results/`")

            if btn:
                name = (st.session_state.get("landing_name") or "").strip()
                if not name:
                    st.warning("Please enter your name to begin.")
                else:
                    st.session_state.annotator_name = name
                    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rng = random.Random(name)
                    order = list(range(TOTAL_PAIRS))
                    rng.shuffle(order)
                    st.session_state.pair_order = order
                    st.session_state.current_idx = 0
                    st.session_state.responses = {}
                    st.session_state.pair_start_time = time.time()
                    st.session_state.study_started = True
                    st.rerun()

    st.stop()

# ─────────────────────────────────────────────
# COMPLETION SCREEN
# ─────────────────────────────────────────────

if st.session_state.current_idx >= TOTAL_PAIRS:
    _, centre, _ = st.columns([0.3, 3.4, 0.3])
    with centre:
        st.markdown("## All done, thank you!")
        st.markdown(
            f"You completed all **{TOTAL_PAIRS}** annotation pairs.  \n"
            "Your responses have been saved automatically."
        )
        annotator = st.session_state.annotator_name.replace(" ", "_")
        saved_path = RESULTS_DIR / f"{annotator}_{st.session_state.session_id}.csv"
        if saved_path.exists():
            final_df = pd.read_csv(saved_path)
            st.markdown(f"Saved to `{saved_path}` — **{len(final_df)} responses**")
            with st.expander("View your responses"):
                st.dataframe(final_df, use_container_width=True)
    st.stop()

# ─────────────────────────────────────────────
# MAIN ANNOTATION VIEW
# ─────────────────────────────────────────────

if st.session_state.scroll_to_top:
    st.session_state.scroll_to_top = False
    components.html(
        """<script>
        var el = window.parent.document.querySelector('section[data-testid="stMain"]');
        if (el) el.scrollTo({top: 0, behavior: 'instant'});
        </script>""",
        height=0,
    )

idx_in_order = st.session_state.pair_order[st.session_state.current_idx]
pair = pairs_df.iloc[idx_in_order]
pair_id = str(pair["pair_id"])

completed = len(st.session_state.responses)
persona_code = persona_short(str(pair["persona_name"]))
persona_icon = PERSONA_ICONS.get(persona_code, "🤖")

# ── Header row ──────────────────────────────
top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown(
        f'<span class="step-pill">Pair {st.session_state.current_idx + 1} of {TOTAL_PAIRS}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"### {persona_icon} {pair['persona_name']}"
    )
with top_right:
    pct = completed / TOTAL_PAIRS
    st.markdown("")
    st.progress(pct, text=f"{completed}/{TOTAL_PAIRS} done")

# ── Mandate box ─────────────────────────────
st.markdown(
    f'<div class="mandate-box"><strong>Mandate:</strong> {pair["mandate_description"]}</div>',
    unsafe_allow_html=True,
)
st.markdown("")

# ── Rationale cards ─────────────────────────
col_a, col_b = st.columns(2, gap="large")

with col_a:
    action_a = str(pair["action_A"]).upper()
    st.markdown(
        f'<div class="card-header">Rationale A</div>'
        f'<span class="action-badge">{action_display(action_a)}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="rcard">{pair["rationale_A"]}</div>',
        unsafe_allow_html=True,
    )

with col_b:
    action_b = str(pair["action_B"]).upper()
    st.markdown(
        f'<div class="card-header">Rationale B</div>'
        f'<span class="action-badge">{action_display(action_b)}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="rcard">{pair["rationale_B"]}</div>',
        unsafe_allow_html=True,
    )

st.markdown("")
st.divider()

# ── Questions ───────────────────────────────
q_col1, q_col2 = st.columns(2, gap="large")

with q_col1:
    st.markdown(
        '<div class="q-label">Q1: Mandate Consistency</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Which rationale better reflects the agent's investment mandate, "
        "considering its stated risk tolerance, asset focus, and decision-making approach?"
    )
    q1 = st.radio(
        "Which rationale is **more consistent** with the agent's stated mandate?",
        options=["Rationale A", "Rationale B", "Both equally", "Neither"],
        index=None,
        key=f"q1_{pair_id}",
        label_visibility="collapsed",
    )
    if q1:
        st.markdown(
            f'<p style="font-size:0.8rem;color:#3d9970;margin:4px 0 0 0;">&#10003;&nbsp;Selected: <strong>{q1}</strong></p>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Select one")

with q_col2:
    st.markdown(
        '<div class="q-label">Q2: Mandate Reminder</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "One agent was shown its mandate again immediately before this decision. "
        "Which rationale shows clearer signs of this: closer alignment with mandate language "
        "or more explicit reference to its stated constraints?"
    )
    q2 = st.radio(
        "Which agent had **just been reminded** of its mandate?",
        options=["Rationale A", "Rationale B", "Cannot tell"],
        index=None,
        key=f"q2_{pair_id}",
        label_visibility="collapsed",
    )
    if q2:
        st.markdown(
            f'<p style="font-size:0.8rem;color:#3d9970;margin:4px 0 0 0;">&#10003;&nbsp;Selected: <strong>{q2}</strong></p>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Select one")

st.markdown("")
comments = st.text_input(
    "💬 Optional comments (anything notable about this pair):",
    key=f"comments_{pair_id}",
    placeholder="Leave blank if nothing to add…",
)

st.markdown("")
both_answered = q1 is not None and q2 is not None

btn_col, _ = st.columns([1, 3])
with btn_col:
    next_clicked = st.button(
        "Next Pair",
        disabled=not both_answered,
        type="primary",
        use_container_width=True,
        key=f"next_{pair_id}",
    )

if next_clicked and both_answered:
    elapsed = round(time.time() - st.session_state.pair_start_time, 1)
    st.session_state.responses[pair_id] = {
        "q1": q1,
        "q2": q2,
        "comments": comments,
        "time_taken": elapsed,
    }
    save_response_incremental(pair_id, q1, q2, comments, elapsed)
    st.session_state.current_idx += 1
    st.session_state.pair_start_time = time.time()
    st.session_state.scroll_to_top = True
    st.rerun()
