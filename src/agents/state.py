"""Shared state threaded through the LangGraph pipeline."""

from typing import TypedDict

from src.schemas import EligibilityAssessment, SourceChunk, UserProfile


class NavigatorState(TypedDict, total=False):
    user_input: str
    profile: UserProfile
    retrieved_chunks: list[SourceChunk]
    assessments: list[EligibilityAssessment]
    draft_answer: str
    final_answer: str
    verification_flags: list[str]
    trace: list[str]
