"""Agent 1: parse the user's free-text situation into a structured UserProfile."""

from src.agents.state import NavigatorState
from src.llm import get_llm
from src.schemas import UserProfile

SYSTEM_PROMPT = """You are the intake agent for a EU social-benefits navigator.
Extract structured facts from the user's free-text description of their situation.

Return ONLY a JSON object with exactly these fields (use null for anything not stated or not inferable — never guess):
{
  "language": "ISO 639-1 code of the language the user wrote in, e.g. en, de, fr, pl, pt, nl, es, it",
  "country": "the EU/EEA country the user lives in, if mentioned or clearly inferable",
  "age": integer or null,
  "employment_status": one of "employed", "unemployed", "retired", "student", "self-employed", "unable_to_work", or null,
  "income_band": one of "very_low", "low", "medium", "high", or null,
  "household": short free-text, e.g. "single", "single parent with 1 child", "couple, no children",
  "disability": true, false, or null,
  "residency_status": one of "citizen", "eu_citizen", "non_eu_permit", "refugee", "undocumented", or null,
  "notes": any other relevant detail the user mentioned (short free text, or null)
}
"""


def run_intake(user_input: str) -> UserProfile:
    llm = get_llm()
    data = llm.complete_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        role="fast",
        temperature=0.0,
    )
    data["raw_situation"] = user_input
    data.setdefault("language", "en")
    return UserProfile(**data)


def intake_node(state: NavigatorState) -> dict:
    profile = run_intake(state["user_input"])
    trace = state.get("trace", []) + [
        f"**Intake Agent** — parsed profile: country={profile.country or 'unknown'}, "
        f"age={profile.age or 'unknown'}, employment={profile.employment_status or 'unknown'}, "
        f"household={profile.household or 'unknown'}, language={profile.language}"
    ]
    return {"profile": profile, "trace": trace}
