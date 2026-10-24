import argparse
import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from lib.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a movie search query using an image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query to rewrite",
    )

    args = parser.parse_args()

    if not args.query.strip():
        parser.error("--query must not be empty")

    image_path = Path(args.image)
    if not image_path.is_file():
        parser.error(f"Image file not found: {image_path}")

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        img = f.read()

    if not img:
        parser.error("Image file is empty")

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set"
        )

    system_prompt = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""

    encoded_image = base64.b64encode(img).decode("ascii")
    data_url = f"data:{mime};base64,{encoded_image}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt.strip(),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {
                    "type": "text",
                    "text": args.query.strip(),
                },
            ],
        }
    ]

    with OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    ) as client:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
        )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("The model returned an empty rewritten query.")

    print(f"Rewritten query: {content.strip()}")

    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()