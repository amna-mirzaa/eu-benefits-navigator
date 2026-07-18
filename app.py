"""Streamlit UI for the EU Benefits Navigator — styled as an official digital service."""

import streamlit as st

from src import config
from src.agents.graph import run_navigator_stream

st.set_page_config(page_title="EU Benefits Navigator", page_icon="🇪🇺", layout="wide")

# ---------------------------------------------------------------------------
# Design system: official "digital public service" look — restrained colour,
# a clear masthead/phase-banner pattern, status tags, and card components.
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;700;800&display=swap');

:root {
    --color-primary: #003399;
    --color-primary-dark: #002266;
    --color-accent: #FFCC00;
    --color-text: #0B0C0C;
    --color-bg-alt: #F3F2F1;
    --color-border: #B1B4B6;
    --color-eligible: #00703C;
    --color-possibly: #855400;
    --color-possibly-bg: #FFF3CD;
    --color-not-eligible: #D4351C;
    --color-insufficient: #505A5F;
}

html, body, [class*="css"], .stApp {
    font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: var(--color-text);
}

h1, h2, h3, h4 { font-weight: 800 !important; letter-spacing: -0.01em; }

/* Masthead */
.eu-masthead {
    background: var(--color-primary);
    color: #fff;
    padding: 1.1rem 1.5rem;
    border-radius: 4px;
    border-bottom: 4px solid var(--color-accent);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.eu-masthead .flag { font-size: 2rem; line-height: 1; }
.eu-masthead .wordmark { font-size: 1.3rem; font-weight: 800; margin: 0; }
.eu-masthead .tagline { font-size: 0.85rem; opacity: 0.85; margin: 0; }

/* Phase banner (govuk-style "this is a prototype" notice) */
.eu-phase-banner {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.6rem 1rem;
    background: var(--color-bg-alt);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
}
.eu-phase-banner .pill {
    background: var(--color-accent);
    color: #1a1a00;
    font-weight: 800;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
    padding: 0.15rem 0.55rem;
    border-radius: 3px;
    text-transform: uppercase;
    flex-shrink: 0;
}

/* Status tags for eligibility verdicts */
.eu-tag {
    display: inline-block;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 0.22rem 0.6rem;
    border-radius: 3px;
    color: #fff;
}
.eu-tag--eligible { background: var(--color-eligible); }
.eu-tag--possibly { background: var(--color-possibly); color: #fff; }
.eu-tag--not-eligible { background: var(--color-not-eligible); }
.eu-tag--insufficient { background: var(--color-insufficient); }

/* Status pills for the sidebar setup checklist */
.eu-status-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; font-size: 0.88rem; }
.eu-status-dot { width: 0.6rem; height: 0.6rem; border-radius: 50%; flex-shrink: 0; }
.eu-status-dot--ok { background: var(--color-eligible); }
.eu-status-dot--missing { background: var(--color-not-eligible); }

/* Answer / result cards */
.eu-card {
    border: 1px solid var(--color-border);
    border-left: 5px solid var(--color-primary);
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    background: #fff;
}
.eu-scheme-card {
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    background: #fff;
}
.eu-scheme-card__header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
}
.eu-scheme-card__title { font-weight: 700; font-size: 1rem; margin: 0; }

/* Buttons */
.stButton > button[kind="primary"] {
    background: var(--color-primary) !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 4px !important;
}
.stButton > button[kind="primary"]:hover { background: var(--color-primary-dark) !important; }
.stButton > button:not([kind="primary"]) {
    border-radius: 4px !important;
    font-size: 0.85rem !important;
}

/* Footer */
.eu-footer {
    margin-top: 2.5rem;
    padding: 1.25rem 1.5rem;
    background: var(--color-bg-alt);
    border-top: 4px solid var(--color-primary);
    border-radius: 4px;
    font-size: 0.82rem;
    color: #3d3d3d;
}
.eu-footer a { color: var(--color-primary); font-weight: 600; }

/* Accessible focus states */
a:focus, button:focus, textarea:focus { outline: 3px solid var(--color-accent) !important; outline-offset: 1px; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Masthead + phase banner
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="eu-masthead">
        <div class="flag">🇪🇺</div>
        <div>
            <p class="wordmark">EU Benefits Navigator</p>
            <p class="tagline">Multi-agent AI · Retrieval-Augmented Generation · Pinecone</p>
        </div>
    </div>
    <div class="eu-phase-banner">
        <span class="pill">Prototype</span>
        <span>
            This is an independent portfolio project, <strong>not an official EU or government service</strong>,
            and does not provide legal or financial advice. See the disclaimer below before relying on any answer.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### About this service")
    st.caption(
        "Helps people discover social benefits they may be entitled to but haven't "
        "claimed — a genuine, under-served EU problem: studies estimate billions of "
        "euros in benefits go unclaimed each year because eligibility rules are "
        "fragmented and hard to navigate."
    )

    st.markdown("### Connection status")
    groq_ok = config.has_groq_key()
    pinecone_ok = config.has_pinecone_key()
    st.markdown(
        f"""
        <div class="eu-status-row">
            <div class="eu-status-dot {'eu-status-dot--ok' if groq_ok else 'eu-status-dot--missing'}"></div>
            <div>Groq (LLM inference) — {'connected' if groq_ok else 'add GROQ_API_KEY to .env'}</div>
        </div>
        <div class="eu-status-row">
            <div class="eu-status-dot {'eu-status-dot--ok' if pinecone_ok else 'eu-status-dot--missing'}"></div>
            <div>Pinecone (vector search) — {'connected' if pinecone_ok else 'add PINECONE_API_KEY to .env'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How it works")
    st.markdown(
        "1. **Intake Agent** — parses your situation into structured facts\n"
        "2. **Retrieval Agent** — RAG search over real EU benefit-scheme documents in Pinecone\n"
        "3. **Eligibility Reasoning Agent** — step-by-step chain-of-thought per scheme "
        "(`openai/gpt-oss-120b` via Groq)\n"
        "4. **Explainer Agent** — writes a plain-language answer\n"
        "5. **Verifier Agent** — cross-checks every claim against the retrieved sources"
    )

    st.markdown("### Coverage")
    st.caption("12 real benefit schemes across Germany, France, Poland, Portugal, Netherlands, Ireland, Spain, and Italy.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Find social benefits you may be entitled to")
st.markdown(
    "Describe your situation in your own words — any EU language works. "
    "The system will search real, published benefit schemes and reason through "
    "your eligibility step by step."
)

EXAMPLES = [
    ("🇩🇪 Retired pensioner", "I'm 68, retired in Germany, living alone on a low pension. My rent is getting hard to afford and I'm worried about heating costs too."),
    ("🇫🇷 Unemployed parent", "I'm 29, unemployed in France, single parent with one 4-year-old child, and I have very little savings left."),
    ("🇳🇱 Low-income renter", "I'm 24, working part-time in the Netherlands, renting a small apartment on my own, and my income is quite low."),
]

def _fill_example(text: str) -> None:
    st.session_state["situation_text"] = text


st.caption("Not sure where to start? Try an example:")
example_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(example_cols, EXAMPLES):
    col.button(label, use_container_width=True, on_click=_fill_example, args=(text,))

user_input = st.text_area(
    "Your situation",
    key="situation_text",
    placeholder="e.g. I'm 68, retired in Germany, living alone on a low pension, and my rent is getting hard to afford.",
    height=120,
)
submitted = st.button("Find benefits", type="primary")

# ---------------------------------------------------------------------------
# Pipeline run with live, step-by-step progress
# ---------------------------------------------------------------------------

STEP_LABELS = {
    "intake": "Intake Agent — reading your situation",
    "retrieve": "Retrieval Agent — searching official benefit-scheme documents",
    "assess": "Eligibility Reasoning Agent — reasoning step-by-step",
    "explain": "Explainer Agent — writing your answer",
    "verify": "Verifier Agent — checking every claim against the sources",
}

VERDICT_STYLES = {
    "Eligible": "eu-tag--eligible",
    "Possibly Eligible": "eu-tag--possibly",
    "Not Eligible": "eu-tag--not-eligible",
    "Insufficient Info": "eu-tag--insufficient",
}


def render_results(result: dict) -> None:
    st.markdown("## Answer")
    st.markdown(
        f'<div class="eu-card">{result.get("final_answer") or "<em>No answer produced.</em>"}</div>',
        unsafe_allow_html=True,
    )

    flags = result.get("verification_flags") or []
    if flags:
        with st.expander("⚑ Verifier flags — claims softened or removed"):
            for flag in flags:
                st.markdown(f"- {flag}")

    assessments = result.get("assessments", [])
    if assessments:
        st.markdown("## Per-scheme eligibility reasoning")
        for a in assessments:
            tag_class = VERDICT_STYLES.get(a.verdict.value, "eu-tag--insufficient")
            st.markdown(
                f"""
                <div class="eu-scheme-card">
                    <div class="eu-scheme-card__header">
                        <span class="eu-tag {tag_class}">{a.verdict.value}</span>
                        <span class="eu-scheme-card__title">{a.scheme_name} ({a.country})</span>
                    </div>
                    <div>{a.justification}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"View reasoning steps — {a.scheme_name}"):
                st.text(a.reasoning_trace)
                if a.citations:
                    st.markdown("**Sources:**")
                    for url in a.citations:
                        st.markdown(f"- [{url}]({url})")

    with st.expander("Agent reasoning trace (full pipeline log)"):
        for line in result.get("trace", []):
            st.markdown(f"- {line}")


if submitted:
    if not user_input.strip():
        st.error("Please describe your situation first.")
    elif not (config.has_groq_key() and config.has_pinecone_key()):
        st.error("Missing API keys — check the connection status in the sidebar and add them to your `.env` file.")
    else:
        result = None
        with st.status("Running the multi-agent pipeline…", expanded=True) as status:
            try:
                for node_name, state in run_navigator_stream(user_input):
                    st.write(f"✅ {STEP_LABELS.get(node_name, node_name)}")
                    result = state
            except Exception as exc:  # surfaced directly for a demo/portfolio app
                status.update(label="Something went wrong", state="error")
                st.exception(exc)
                result = None
            else:
                status.update(label="Pipeline complete", state="complete", expanded=False)

        if result:
            render_results(result)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="eu-footer">
        <strong>Disclaimer:</strong> This tool provides general, AI-generated guidance based on a sample of
        publicly available benefit-scheme summaries. It is not legal, tax, or financial advice, may not reflect
        the latest rule changes, and does not cover every scheme or country. Always verify eligibility and
        current amounts on the relevant official government website before applying.
        <br><br>
        Portfolio project · <a href="https://github.com/amna-mirzaa/eu-benefits-navigator" target="_blank">Source code on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
