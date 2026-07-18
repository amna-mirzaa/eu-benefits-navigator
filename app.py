"""Streamlit demo UI for the EU Benefits Navigator."""

import streamlit as st

from src import config
from src.agents.graph import run_navigator

st.set_page_config(page_title="EU Benefits Navigator", page_icon="🇪🇺", layout="wide")

with st.sidebar:
    st.header("🇪🇺 EU Benefits Navigator")
    st.caption(
        "A multi-agent RAG assistant that helps people discover social benefits "
        "they may be entitled to but haven't claimed — a genuine, under-served "
        "EU problem: billions of euros in benefits go unclaimed each year because "
        "eligibility rules are fragmented and hard to navigate."
    )

    st.subheader("Setup status")
    st.write("✅ Groq API key" if config.has_groq_key() else "❌ Groq API key — add `GROQ_API_KEY` to `.env`")
    st.write(
        "✅ Pinecone API key" if config.has_pinecone_key() else "❌ Pinecone API key — add `PINECONE_API_KEY` to `.env`"
    )

    st.subheader("Architecture")
    st.markdown(
        "1. **Intake Agent** — parses your situation into structured facts\n"
        "2. **Retrieval Agent** — RAG search over real EU benefit-scheme documents in Pinecone\n"
        "3. **Eligibility Reasoning Agent** — step-by-step chain-of-thought per scheme "
        "(openai/gpt-oss-120b via Groq)\n"
        "4. **Explainer Agent** — writes a plain-language answer\n"
        "5. **Verifier Agent** — cross-checks every claim against the retrieved sources"
    )

    st.warning(
        "⚠️ **Not legal or financial advice.** This is a portfolio project covering a "
        "sample of real schemes in a handful of countries. Always confirm eligibility "
        "and current thresholds on the official government page before applying.",
        icon="⚠️",
    )

st.title("Find social benefits you may be entitled to")
st.caption("Describe your situation in your own words — any EU language works.")

example = "I'm 68, retired in Germany, living alone on a low pension, and my rent is getting hard to afford."
user_input = st.text_area("Your situation", placeholder=example, height=120)
submitted = st.button("Find benefits", type="primary")

if submitted:
    if not user_input.strip():
        st.error("Please describe your situation first.")
    elif not (config.has_groq_key() and config.has_pinecone_key()):
        st.error("Missing API keys — check the setup status in the sidebar and add them to your `.env` file.")
    else:
        with st.spinner("Running the multi-agent pipeline (intake → retrieve → reason → explain → verify)…"):
            try:
                result = run_navigator(user_input)
            except Exception as exc:  # surfaced directly for a demo/portfolio app
                st.exception(exc)
                result = None

        if result:
            st.subheader("Answer")
            st.markdown(result.get("final_answer") or "_No answer produced._")

            flags = result.get("verification_flags") or []
            if flags:
                with st.expander("⚑ Verifier flags (claims softened or removed)"):
                    for flag in flags:
                        st.markdown(f"- {flag}")

            st.divider()
            st.subheader("Agent reasoning trace")
            for line in result.get("trace", []):
                st.markdown(f"- {line}")

            assessments = result.get("assessments", [])
            if assessments:
                st.subheader("Per-scheme eligibility reasoning")
                for a in assessments:
                    with st.expander(f"{a.verdict.value} — {a.scheme_name} ({a.country})"):
                        st.markdown(f"**Justification:** {a.justification}")
                        st.markdown("**Chain of reasoning:**")
                        st.text(a.reasoning_trace)
                        if a.citations:
                            st.markdown("**Sources:**")
                            for url in a.citations:
                                st.markdown(f"- [{url}]({url})")
