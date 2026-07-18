"""Unit tests using mocked LLM/vectorstore — no real API keys required."""

from unittest.mock import MagicMock, patch

from langgraph.graph.state import CompiledStateGraph

from src.agents import eligibility_agent, intake_agent, retrieval_agent, verifier_agent
from src.agents.graph import build_graph
from src.agents.state import NavigatorState
from src.schemas import EligibilityAssessment, EligibilityVerdict, SourceChunk, UserProfile


def make_chunk(country="Germany", scheme="Wohngeld", text="Some rule text.", score=0.9):
    return SourceChunk(
        chunk_id=f"{country}-{scheme}-0",
        country=country,
        scheme_name=scheme,
        category="housing",
        source_url="https://example.gov/wohngeld",
        text=text,
        score=score,
    )


# ---------------------------------------------------------------------------
# 1. Intake agent: profile parsing shape
# ---------------------------------------------------------------------------


def test_intake_profile_shape():
    fake_llm = MagicMock()
    fake_llm.complete_json.return_value = {
        "language": "en",
        "country": "Germany",
        "age": 68,
        "employment_status": "retired",
        "income_band": "low",
        "household": "single",
        "disability": None,
        "residency_status": "citizen",
        "notes": "low pension, struggling with rent",
    }

    with patch.object(intake_agent, "get_llm", return_value=fake_llm):
        profile = intake_agent.run_intake(
            "I'm 68, retired in Germany, living alone on a low pension, rent is hard to afford."
        )

    assert isinstance(profile, UserProfile)
    assert profile.country == "Germany"
    assert profile.age == 68
    assert profile.employment_status == "retired"
    assert profile.language == "en"
    assert profile.raw_situation.startswith("I'm 68")


def test_intake_node_updates_trace():
    fake_llm = MagicMock()
    fake_llm.complete_json.return_value = {"language": "en", "country": "France"}

    with patch.object(intake_agent, "get_llm", return_value=fake_llm):
        state: NavigatorState = {"user_input": "I live in France and lost my job.", "trace": []}
        update = intake_agent.intake_node(state)

    assert "profile" in update
    assert len(update["trace"]) == 1
    assert "Intake Agent" in update["trace"][0]


# ---------------------------------------------------------------------------
# 2. Graph wiring / state threading
# ---------------------------------------------------------------------------


def test_graph_compiles_with_expected_nodes():
    compiled = build_graph()
    assert isinstance(compiled, CompiledStateGraph)
    node_names = set(compiled.get_graph().nodes.keys())
    assert {"intake", "retrieve", "assess", "explain", "verify"}.issubset(node_names)


def test_state_threads_trace_across_nodes():
    """Trace list should accumulate (not overwrite) as state flows intake -> retrieve."""
    fake_llm = MagicMock()
    fake_llm.complete_json.return_value = {"language": "en", "country": "Germany"}

    chunk = make_chunk()
    with patch.object(intake_agent, "get_llm", return_value=fake_llm), patch.object(
        retrieval_agent, "retrieve_chunks", return_value=[chunk]
    ):
        state: NavigatorState = {"user_input": "retired, low income, Germany", "trace": []}
        state.update(intake_agent.intake_node(state))
        state.update(retrieval_agent.retrieval_node(state))

    assert len(state["trace"]) == 2
    assert "Intake Agent" in state["trace"][0]
    assert "Retrieval Agent" in state["trace"][1]
    assert state["retrieved_chunks"] == [chunk]


# ---------------------------------------------------------------------------
# 3. Verifier agent: strips/flags an unsupported claim
# ---------------------------------------------------------------------------


def test_verifier_strips_unsupported_claim():
    fake_llm = MagicMock()
    fake_llm.complete.return_value = (
        "VERIFIED_ANSWER:\n"
        "You may qualify for Wohngeld. Check the official page for the exact current amount.\n\n"
        "FLAGS:\n"
        "- Removed an invented figure of \"€850/month\" not present in the source text.\n"
    )

    assessment = EligibilityAssessment(
        scheme_name="Wohngeld",
        country="Germany",
        category="housing",
        verdict=EligibilityVerdict.POSSIBLY_ELIGIBLE,
        reasoning_trace="...",
        justification="Low income tenant, may qualify.",
        citations=["https://example.gov/wohngeld"],
        source_chunks=[make_chunk()],
    )

    with patch.object(verifier_agent, "get_llm", return_value=fake_llm):
        verified, flags = verifier_agent.run_verifier(
            draft_answer="You may qualify for Wohngeld, worth about €850/month.",
            assessments=[assessment],
        )

    assert "€850" not in verified
    assert len(flags) == 1
    assert "€850" in flags[0]


def test_verifier_no_flags_when_none_reported():
    fake_llm = MagicMock()
    fake_llm.complete.return_value = "VERIFIED_ANSWER:\nAll good.\n\nFLAGS:\n- None\n"

    with patch.object(verifier_agent, "get_llm", return_value=fake_llm):
        verified, flags = verifier_agent.run_verifier("All good.", [])

    assert verified == "All good."
    assert flags == []


# ---------------------------------------------------------------------------
# Eligibility agent: FINAL_JSON parsing
# ---------------------------------------------------------------------------


def test_eligibility_agent_parses_final_json_and_reasoning():
    fake_llm = MagicMock()
    fake_llm.complete_with_reasoning.return_value = (
        "The user is 68 and retired, income is low, rent is a burden. This matches the "
        "low-income renter criterion.\n"
        'FINAL_JSON: {"verdict": "Possibly Eligible", "justification": "Low pension renter likely qualifies."}',
        "Step 1: check age -> not a criterion. Step 2: check income -> low, matches. Step 3: check tenancy -> renter, matches.",
    )

    profile = UserProfile(raw_situation="68, retired, low pension, renting", country="Germany", age=68)
    chunk = make_chunk()

    with patch.object(eligibility_agent, "get_llm", return_value=fake_llm):
        assessment = eligibility_agent.assess_scheme(
            profile, "Germany", "Wohngeld", "housing", [chunk]
        )

    assert assessment.verdict == EligibilityVerdict.POSSIBLY_ELIGIBLE
    assert "Low pension renter" in assessment.justification
    assert "Step 1" in assessment.reasoning_trace
    assert assessment.citations == ["https://example.gov/wohngeld"]


def test_eligibility_agent_tolerates_markdown_wrapped_final_json():
    """Regression test: some models bold the marker as **FINAL_JSON:** — the
    parser must still find the JSON object rather than silently falling back
    to raw text (this was a real bug found against the live Groq API)."""
    fake_llm = MagicMock()
    fake_llm.complete_with_reasoning.return_value = (
        "Some closing remark.\n\n"
        '**FINAL_JSON:** {"verdict": "Not Eligible", "justification": "Too old for this scheme."}',
        "Step 1: check age -> fails.",
    )

    profile = UserProfile(raw_situation="68, retired", country="Germany", age=68)
    chunk = make_chunk(scheme="Buergergeld")

    with patch.object(eligibility_agent, "get_llm", return_value=fake_llm):
        assessment = eligibility_agent.assess_scheme(
            profile, "Germany", "Buergergeld", "unemployment_minimum_income", [chunk]
        )

    assert assessment.verdict == EligibilityVerdict.NOT_ELIGIBLE
    assert assessment.justification == "Too old for this scheme."
