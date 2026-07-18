"""Pydantic data models shared across agents."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    raw_situation: str
    language: str = "en"
    country: Optional[str] = None
    age: Optional[int] = None
    employment_status: Optional[str] = None
    income_band: Optional[str] = None
    household: Optional[str] = None
    disability: Optional[bool] = None
    residency_status: Optional[str] = None
    notes: Optional[str] = None


class SourceChunk(BaseModel):
    chunk_id: str
    country: str
    scheme_name: str
    category: str
    source_url: str
    text: str
    score: float = 0.0


class EligibilityVerdict(str, Enum):
    ELIGIBLE = "Eligible"
    POSSIBLY_ELIGIBLE = "Possibly Eligible"
    NOT_ELIGIBLE = "Not Eligible"
    INSUFFICIENT_INFO = "Insufficient Info"


class EligibilityAssessment(BaseModel):
    scheme_name: str
    country: str
    category: str
    verdict: EligibilityVerdict
    reasoning_trace: str = Field(description="Step-by-step chain of reasoning")
    justification: str = Field(description="Short human-readable summary")
    citations: list[str] = Field(default_factory=list)
    source_chunks: list[SourceChunk] = Field(default_factory=list)


class VerifiedClaim(BaseModel):
    claim: str
    supported: bool
    note: Optional[str] = None
