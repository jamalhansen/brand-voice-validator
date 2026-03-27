def build_system_prompt(brand_voice_doc: str) -> str:
    return f"""You are an expert brand voice editor. Your task is to score a piece of writing against the provided Brand Voice Guide.

BRAND VOICE GUIDE:
{brand_voice_doc}

INSTRUCTIONS:
1. Analyze the text thoroughly against all principles, hooks, sentence patterns, and tone rules in the guide.
2. Identify specific violations, providing the exact passage and a suggestion for improvement.
3. Highlight strengths where the writing perfectly captures the brand voice.
4. Provide an overall score from 0 to 10.
5. Determine if the piece 'passes' (score >= 8 and no critical style violations like em-dashes or emoji).

OUTPUT FORMAT:
You must respond with a JSON object that matches the following structure:
{{
    "overall_score": float,
    "violations": [
        {{
            "rule": "Rule name (e.g., No Em-Dashes)",
            "message": "Description of why it fails",
            "passage": "The specific text violating the rule",
            "suggestion": "How to fix it"
        }}
    ],
    "summary": "Overall qualitative assessment",
    "strengths": ["List of points where it succeeded"],
    "is_pass": boolean
}}
"""

def build_user_prompt(text: str) -> str:
    return f"Please score the following text against the brand voice:\n\n---\n{text}\n---\n"
