import re
from local_first_common.tracking import timed_run
from .schema import BrandVoiceScore, RuleViolation


class BrandVoiceValidatorError(Exception):
    """Base error for strict brand voice validator operations."""


class InputFileNotFoundError(BrandVoiceValidatorError):
    """Raised when the input markdown file does not exist."""


class VaultPathMissingError(BrandVoiceValidatorError):
    """Raised when OBSIDIAN_VAULT_PATH is not configured."""


class BrandVoiceFileNotFoundError(BrandVoiceValidatorError):
    """Raised when the brand voice document cannot be found."""


class ProviderResolutionError(BrandVoiceValidatorError):
    """Raised when the configured provider/model cannot be initialized."""


class ScoringError(BrandVoiceValidatorError):
    """Raised when LLM scoring fails."""


# Regex patterns for guardrails
EM_DASH_PATTERN = re.compile(r"[\u2014]")
# Simple emoji pattern - covers most common emojis
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)


def check_python_analogy(text: str) -> bool:
    """Basic check if 'Python' or typical Python concepts are mentioned."""
    keywords = ["python", "def ", "import ", "list comprehension", "f-string"]
    return any(kw in text.lower() for kw in keywords)


def apply_guardrails(text: str, score: BrandVoiceScore) -> BrandVoiceScore:
    """Apply deterministic regex-based rules as a layer on top of LLM results."""

    # 1. No Em-Dashes
    em_dashes = EM_DASH_PATTERN.findall(text)
    if em_dashes:
        score.violations.append(
            RuleViolation(
                rule="No Em-Dashes",
                message=f"Found {len(em_dashes)} em-dashes. Use commas, periods, or semicolons instead.",
                passage="Contains \u2014",
                suggestion="Replace \u2014 with alternative punctuation.",
            )
        )
        score.is_pass = False

    # 2. No Emoji
    emojis = EMOJI_PATTERN.findall(text)
    if emojis:
        score.violations.append(
            RuleViolation(
                rule="No Emoji",
                message=f"Found {len(emojis)} emoji. Emojis are banned in blog posts and newsletters.",
                passage="".join(emojis[:5]),
                suggestion="Remove emojis.",
            )
        )
        score.is_pass = False

    # 3. Python Analogy
    if not check_python_analogy(text):
        score.violations.append(
            RuleViolation(
                rule="Python Analogy Missing",
                message="No Python references found. Teaching starts from what the reader knows (Python).",
                suggestion="Include a Python code comparison or analogy.",
            )
        )
        if score.overall_score > 7:
            score.overall_score -= 1.0

    return score


def _score_or_raise(
    llm, system: str, user: str, source_location: str, text_to_score: str
) -> BrandVoiceScore:
    try:
        with timed_run(
            "brand-voice-validator", llm.model, source_location=source_location
        ) as run:
            response = llm.complete(system, user, response_model=BrandVoiceScore)
            result = apply_guardrails(text_to_score, response)
            run.item_count = 1
            return result
    except Exception as e:  # noqa: BLE001
        raise ScoringError(str(e)) from e
