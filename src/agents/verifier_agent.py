"""Agent 5: self-verification — strip or flag any claim not grounded in retrieved sources."""

import re

from src.agents.state import NavigatorState
from src.llm import get_llm
from src.schemas import EligibilityAssessment

SYSTEM_PROMPT = """You are the verifier agent for a EU social-benefits navigator.
You will be given a draft answer and the official source excerpts it was supposed to be
grounded in. Your job is quality control, not rewriting style.

Cross-check every specific factual claim in the draft (amounts, thresholds, ages, deadlines,
eligibility conditions) against the source excerpts:
- If a claim is directly supported, keep it as-is.
- If a claim is NOT supported by the excerpts (invented, or over-specific), remove or soften
  it (e.g. replace a precise unsupported number with "check the official page for the exact
  current amount").
- Do not add new facts.

Output exactly two sections:
VERIFIED_ANSWER:
<the corrected final answer, same language as the draft>

FLAGS:
- <short bullet for each claim you removed or softened, or a single line "None" if nothing needed changing>
"""

SPLIT_RE = re.compile(r"VERIFIED_ANSWER:\s*(.*?)\s*FLAGS:\s*(.*)", re.DOTALL)


def _source_excerpts(assessments: list[EligibilityAssessment]) -> str:
    parts = []
    for a in assessments:
        for chunk in a.source_chunks:
            parts.append(f"[{chunk.country} – {chunk.scheme_name}] {chunk.text}")
    return "\n\n".join(parts) if parts else "No source excerpts available."


def run_verifier(draft_answer: str, assessments: list[EligibilityAssessment]) -> tuple[str, list[str]]:
    llm = get_llm()
    user_content = (
        f"DRAFT ANSWER:\n{draft_answer}\n\n"
        f"SOURCE EXCERPTS:\n{_source_excerpts(assessments)}"
    )
    result = llm.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        role="fast",
        temperature=0.0,
    )

    match = SPLIT_RE.search(result)
    if not match:
        return draft_answer, ["Verifier could not parse its output; returning unverified draft."]

    verified_answer, flags_block = match.groups()
    flags = [
        line.strip("- ").strip()
        for line in flags_block.strip().splitlines()
        if line.strip() and line.strip().lower() not in {"none", "- none"}
    ]
    return verified_answer.strip(), flags


def verifier_node(state: NavigatorState) -> dict:
    verified, flags = run_verifier(state.get("draft_answer", ""), state.get("assessments", []))
    trace = state.get("trace", []) + [
        f"**Verifier Agent** — cross-checked the draft against retrieved sources; "
        f"{len(flags)} claim(s) flagged/softened" if flags else
        "**Verifier Agent** — cross-checked the draft against retrieved sources; all claims supported"
    ]
    return {"final_answer": verified, "verification_flags": flags, "trace": trace}
