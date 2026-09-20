
from pydantic import BaseModel, Field


class RuleViolation(BaseModel):
    rule: str
    message: str
    passage: str | None = None
    suggestion: str | None = None

class BrandVoiceScore(BaseModel):
    overall_score: float = Field(..., ge=0, le=10, description="Score from 0 to 10")
    violations: list[RuleViolation]
    summary: str
    strengths: list[str]
    is_pass: bool
