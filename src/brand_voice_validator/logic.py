import logging
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from local_first_common.providers import PROVIDERS
from local_first_common.cli import (
    init_config_option,
    provider_option,
    model_option,
    dry_run_option,
    no_llm_option,
    verbose_option,
    debug_option,
    pipe_option,
    json_option,
    resolve_provider,
    resolve_dry_run,
)
from local_first_common.logging import setup_logging
from local_first_common.config import get_setting
from local_first_common.tracking import register_tool, timed_run

from .schema import BrandVoiceScore, RuleViolation
from .prompts import build_system_prompt, build_user_prompt


TOOL_NAME = "brand-voice-validator"
DEFAULTS = {"provider": "ollama", "model": "llama3"}
_TOOL = register_tool("brand-voice-validator")
console = Console()
app = typer.Typer(help="Scores a piece of writing against brand voice rules.")


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
        # This might not be a hard 'fail' but definitely a deduction
        if score.overall_score > 7:
            score.overall_score -= 1.0

    return score


def display_score(score: BrandVoiceScore, out_console: Optional[Console] = None):
    """Rich display of the score result."""
    target_console = out_console or console
    color = "green" if score.is_pass else "red"
    target_console.print(
        f"\n[bold {color}]OVERALL SCORE: {score.overall_score}/10[/bold {color}]"
    )
    target_console.print(
        f"[bold]Pass:[/bold] {'[green]YES[/green]' if score.is_pass else '[red]NO[/red]'}"
    )
    target_console.print(f"\n[bold]Summary:[/bold]\n{score.summary}")

    if score.strengths:
        target_console.print("\n[bold green]Strengths:[/bold green]")
        for strength in score.strengths:
            target_console.print(f"  • {strength}")

    if score.violations:
        target_console.print("\n[bold red]Violations:[/bold red]")
        table = Table(show_header=True, header_style="bold red")
        table.add_column("Rule", style="dim", width=20)
        table.add_column("Message")
        table.add_column("Passage", style="italic")

        for v in score.violations:
            table.add_row(v.rule, v.message, v.passage or "N/A")
        target_console.print(table)


def _resolve_paths_or_raise(
    input_file: Optional[Path] = None, is_pipe: bool = False
) -> tuple[str, str, Path]:
    if is_pipe:
        text_to_score = sys.stdin.read()
        source_location = "stdin"
    else:
        if input_file is None or not input_file.exists():
            raise InputFileNotFoundError(f"File {input_file} not found.")
        text_to_score = input_file.read_text(encoding="utf-8")
        source_location = str(input_file)

    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        raise VaultPathMissingError("OBSIDIAN_VAULT_PATH environment variable not set.")

    brand_voice_file = Path(vault_path) / "brand" / "_BRAND_VOICE.md"
    if not brand_voice_file.exists():
        raise BrandVoiceFileNotFoundError(
            f"Brand voice file not found at {brand_voice_file}"
        )

    return text_to_score, source_location, brand_voice_file


def _resolve_llm_or_raise(
    provider: str,
    model: Optional[str],
    debug: bool,
    no_llm: bool,
):
    actual_provider = get_setting(
        TOOL_NAME, "provider", cli_val=provider, default="ollama"
    )
    actual_model = get_setting(TOOL_NAME, "model", cli_val=model)
    try:
        return resolve_provider(
            PROVIDERS, actual_provider, actual_model, debug=debug, no_llm=no_llm
        )
    except Exception as e:  # noqa: BLE001
        raise ProviderResolutionError(str(e)) from e


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


@app.command()
def score(
    path: Annotated[
        Optional[Path],
        typer.Argument(help="Optional input file (or '-' for stdin)."),
    ] = None,
    input_file: Annotated[
        Optional[Path],
        typer.Option("--input", "-i", help="Input file path."),
    ] = None,
    pipe: Annotated[bool, pipe_option()] = False,
    json_output: Annotated[bool, json_option()] = False,
    provider: Annotated[str, provider_option(PROVIDERS)] = os.environ.get(
        "MODEL_PROVIDER", "ollama"
    ),
    model: Annotated[Optional[str], model_option()] = None,
    dry_run: Annotated[bool, dry_run_option()] = False,
    no_llm: Annotated[bool, no_llm_option()] = False,
    verbose: Annotated[bool, verbose_option()] = False,
    debug: Annotated[bool, debug_option()] = False,
    init_config: Annotated[bool, init_config_option(TOOL_NAME, DEFAULTS)] = False,
):
    """Score a file against brand voice."""
    dry_run = resolve_dry_run(dry_run, no_llm)

    log_level = (
        logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    )
    setup_logging(level=log_level, tool_name=TOOL_NAME, persist_warnings=True)

    target_file = input_file or path
    is_pipe = pipe or (target_file is not None and str(target_file) == "-")

    if not is_pipe and target_file is None:
        console.print(
            "[red]Error: Must provide an input file (--input/-i) or pipe via stdin.[/red]"
        )
        raise typer.Exit(1)

    try:
        text_to_score, source_location, brand_voice_file = _resolve_paths_or_raise(
            target_file, is_pipe=is_pipe
        )
    except (
        InputFileNotFoundError,
        VaultPathMissingError,
        BrandVoiceFileNotFoundError,
    ) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    brand_voice_doc = brand_voice_file.read_text(encoding="utf-8")

    try:
        llm = _resolve_llm_or_raise(provider, model, debug, no_llm)
    except ProviderResolutionError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    system = build_system_prompt(brand_voice_doc)
    user = build_user_prompt(text_to_score)

    if verbose:
        console.print(f"Scoring {source_location} using {llm.model}...")

    try:
        result = _score_or_raise(llm, system, user, source_location, text_to_score)
    except ScoringError as e:
        console.print(f"[red]Error during processing: {e}[/red]")
        raise typer.Exit(1)

    if json_output:
        print(result.model_dump_json(indent=2))
        if not result.is_pass:
            raise typer.Exit(1)
        return

    if is_pipe:
        err_console = Console(stderr=True)
        display_score(result, out_console=err_console)
        if result.is_pass:
            sys.stdout.write(text_to_score)
        else:
            raise typer.Exit(1)
        return

    display_score(result)

    if dry_run:
        console.print(
            "\n[yellow][dry-run] Analysis complete. No results persisted.[/yellow]"
        )
    else:
        pass



if __name__ == "__main__":
    app()
