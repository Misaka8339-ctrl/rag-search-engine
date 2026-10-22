import os

from dotenv import load_dotenv
from openai import OpenAI

from .config import PROJECT_ROOT


def _enhance_with_llm(prompt: str) -> str:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set"
        )

    with OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    ) as client:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("The model returned an empty enhanced query.")

    return content.strip()


def correct_spelling(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    return _enhance_with_llm(prompt)

def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    return _enhance_with_llm(prompt)

def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.
Do not repeat words already present in the original query.

Examples:
- "scary bear movie" -> "horror grizzly terrifying film"
- "action movie with bear" -> "thriller chase fight adventure"
- "comedy with bear" -> "funny humor lighthearted"

User query: "{query}"
"""
    additional_terms = _enhance_with_llm(prompt)

    return f"{query} {additional_terms}"