from typing import List, Optional
from pydantic import BaseModel, Field

class RuleViolation(BaseModel):
    rule: str
    message: str
    passage: Optional[str] = None
    suggestion: Optional[str] = None

class BrandVoiceScore(BaseModel):
    overall_score: float = Field(..., ge=0, le=10, description="Score from 0 to 10")
    violations: List[RuleViolation]
    summary: str
    strengths: List[str]
    is_pass: bool
