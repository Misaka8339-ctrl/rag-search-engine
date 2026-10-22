import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

def main() -> None:
    # 明确读取当前脚本所在目录下的 .env
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set"
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": (
                    "Why is Boot.dev such a great place to learn about RAG? "
                    "Use one paragraph maximum."
                ),
            }
        ],
    )

    print(response.choices[0].message.content)

    usage = response.usage
    if usage is None:
        raise RuntimeError("The provider did not return token usage.")

    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Response tokens: {usage.completion_tokens}")


if __name__ == "__main__":
    main()