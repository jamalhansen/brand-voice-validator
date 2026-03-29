from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from brand_voice_validator.logic import app, display_score
from brand_voice_validator.schema import BrandVoiceScore, RuleViolation

runner = CliRunner()

def test_display_score_pass(capsys):
    score = BrandVoiceScore(
        overall_score=9.0,
        violations=[],
        summary="Excellent draft.",
        strengths=["Direct tone", "Clear Python analogy"],
        is_pass=True
    )
    display_score(score)
    captured = capsys.readouterr()
    assert "OVERALL SCORE: 9.0/10" in captured.out
    assert "Pass: YES" in captured.out
    assert "Strengths:" in captured.out
    assert "Direct tone" in captured.out

def test_display_score_fail_with_violations(capsys):
    score = BrandVoiceScore(
        overall_score=4.0,
        violations=[
            RuleViolation(rule="No Emoji", message="Found 🚀", passage="emoji 🚀")
        ],
        summary="Too many emojis.",
        strengths=[],
        is_pass=False
    )
    display_score(score)
    captured = capsys.readouterr()
    assert "OVERALL SCORE: 4.0/10" in captured.out
    assert "Pass: NO" in captured.out
    assert "Violations:" in captured.out
    assert "No Emoji" in captured.out

@patch("brand_voice_validator.logic.resolve_provider")
@patch("brand_voice_validator.logic.timed_run")
@patch("os.getenv")
def test_score_command_success(mock_getenv, mock_timed_run, mock_resolve_provider, tmp_path):
    # Setup mock vault and input file
    vault_path = tmp_path / "vault"
    brand_dir = vault_path / "brand"
    brand_dir.mkdir(parents=True)
    brand_file = brand_dir / "_BRAND_VOICE.md"
    brand_file.write_text("Brand voice rules...")
    
    input_file = tmp_path / "input.md"
    input_file.write_text("Input text for scoring with a Python analogy.")
    
    mock_getenv.return_value = str(vault_path)
    
    # Mock LLM provider
    mock_llm = MagicMock()
    mock_llm.model = "mock-model"
    mock_llm.complete.return_value = BrandVoiceScore(
        overall_score=9.0,
        violations=[],
        summary="Great job!",
        strengths=["Clear"],
        is_pass=True
    )
    mock_resolve_provider.return_value = mock_llm
    
    # Mock timed_run context manager
    mock_timed_run.return_value.__enter__.return_value = MagicMock()

    result = runner.invoke(app, ["--input", str(input_file), "--no-llm"])
    
    assert result.exit_code == 0
    assert "OVERALL SCORE: 9.0/10" in result.stdout

def test_score_command_file_not_found():
    result = runner.invoke(app, ["--input", "nonexistent.md"])
    assert result.exit_code == 1
    assert "Error: File nonexistent.md not found" in result.stdout

@patch("os.getenv")
def test_score_command_no_vault_path(mock_getenv, tmp_path):
    input_file = tmp_path / "input.md"
    input_file.write_text("content")
    mock_getenv.return_value = None
    
    result = runner.invoke(app, ["--input", str(input_file)])
    assert result.exit_code == 1
    assert "Error: OBSIDIAN_VAULT_PATH environment variable not set" in result.stdout
