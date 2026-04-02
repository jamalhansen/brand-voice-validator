# Brand Voice Validator

Scores a piece of writing against the full brand voice document located in your Obsidian vault.

## Features
- **Deterministic Guardrails**: Uses regex to strictly enforce "No Em-Dashes" and "No Emoji" rules.
- **Python Analogy Check**: Ensures teaching starts from familiar Python concepts.
- **LLM Assessment**: Scores tone, rhythm, and teaching patterns against the brand voice guide.
- **Rich Output**: Provides a detailed summary, strengths, and actionable suggestions for violations.

## Installation
```bash
uv sync
```

## Usage
The tool is available as `brand-voice-validator` after `uv sync`.

```bash
export OBSIDIAN_VAULT_PATH="/path/to/your/vault"
uv run brand-voice-validator score -i draft.md
```

Standard flags supported: `--dry-run`, `--no-llm`, `--provider`, `--model`.
