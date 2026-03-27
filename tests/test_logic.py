from brand_voice_validator.logic import apply_guardrails
from brand_voice_validator.schema import BrandVoiceScore

def test_apply_guardrails_em_dash():
    text = "This is a test — with an em-dash."
    score = BrandVoiceScore(
        overall_score=9.0,
        violations=[],
        summary="Good draft.",
        strengths=["Direct tone"],
        is_pass=True
    )
    
    result = apply_guardrails(text, score)
    
    assert result.is_pass is False
    assert any(v.rule == "No Em-Dashes" for v in result.violations)

def test_apply_guardrails_emoji():
    text = "This is a test with emoji 🚀."
    score = BrandVoiceScore(
        overall_score=9.0,
        violations=[],
        summary="Good draft.",
        strengths=["Direct tone"],
        is_pass=True
    )
    
    result = apply_guardrails(text, score)
    
    assert result.is_pass is False
    assert any(v.rule == "No Emoji" for v in result.violations)

def test_apply_guardrails_python_missing():
    text = "No mention of the snake language here."
    score = BrandVoiceScore(
        overall_score=9.0,
        violations=[],
        summary="Good draft.",
        strengths=["Direct tone"],
        is_pass=True
    )
    
    result = apply_guardrails(text, score)
    
    assert result.overall_score == 8.0
    assert any(v.rule == "Python Analogy Missing" for v in result.violations)
