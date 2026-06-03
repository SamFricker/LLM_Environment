import os
from typing import List, Optional

from openai import OpenAI


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def parse_model_list(value: str) -> List[str]:
    models = []
    for model in value.split(","):
        model = model.strip()
        if model and model not in models:
            models.append(model)
    return models


DEFAULT_OPENROUTER_MODEL = "openrouter/free"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL
OPENROUTER_MODELS = parse_model_list(os.getenv("OPENROUTER_MODELS", f"{OPENROUTER_MODEL},openrouter/auto"))


def create_client() -> Optional[OpenAI]:
    if not OPENROUTER_API_KEY:
        return None

    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


client = create_client()


def run_chat():
    if client is None:
        print("Missing OPENROUTER_API_KEY. Set it in your environment before running chat.py.")
        return

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    ]

    print("Chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({
            "role": "user",
            "content": user_input,
        })

        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                temperature=0.2,
            )

            assistant_reply = response.choices[0].message.content

            print(f"\nAssistant: {assistant_reply}\n")

            messages.append({
                "role": "assistant",
                "content": assistant_reply,
            })

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    run_chat()
