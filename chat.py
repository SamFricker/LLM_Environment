import os
from typing import Optional

from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("CODEX_API_KEY")
OPENAI_KEY_SOURCE = "OPENAI_API_KEY" if os.getenv("OPENAI_API_KEY") else "CODEX_API_KEY" if os.getenv("CODEX_API_KEY") else ""
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
OPENROUTER_ROUTE_MODEL = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL
OPENROUTER_MODELS = [OPENROUTER_ROUTE_MODEL]

SYSTEM_INSTRUCTIONS = "You are a helpful assistant. Answer clearly and concisely."


def create_openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


def create_openrouter_client() -> Optional[OpenAI]:
    if not OPENROUTER_API_KEY:
        return None
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


OPENAI_CLIENT = create_openai_client()
OPENROUTER_CLIENT = create_openrouter_client()

# Compatibility for main.py/mainNEW.py imports. Prefer direct OpenAI when available.
client = OPENAI_CLIENT or OPENROUTER_CLIENT
OPENROUTER_MODEL = OPENAI_MODEL if OPENAI_CLIENT is not None else OPENROUTER_ROUTE_MODEL


def active_provider() -> str:
    if OPENAI_CLIENT is not None:
        return "openai"
    if OPENROUTER_CLIENT is not None:
        return "openrouter"
    return "none"


def active_model() -> str:
    return OPENAI_MODEL if active_provider() == "openai" else OPENROUTER_ROUTE_MODEL


def friendly_error(error: Exception) -> str:
    text = str(error)
    lower = text.lower()

    if "invalid_api_key" in lower or "incorrect api key" in lower or "401" in lower:
        return "The API key was rejected. Check that OPENAI_API_KEY is set to a valid OpenAI API key."
    if "insufficient_quota" in lower or "billing" in lower:
        return "The OpenAI account/key does not currently have usable API credits or billing enabled."
    if "rate_limit" in lower or "rate limit" in lower or "429" in lower:
        return "The API rate-limited this request. Wait briefly, or use a model/key with higher limits."
    if "free-models-per-day" in lower:
        return "OpenRouter says the free-model daily quota is exhausted for this account."
    if "503" in lower or "no healthy" in lower or "no available model provider" in lower:
        return "The provider could not serve the selected model right now. Try again later or use a different model."
    return text


def run_openai_chat():
    previous_response_id = None

    print(f"Chat started with OpenAI model {OPENAI_MODEL} using {OPENAI_KEY_SOURCE}. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        try:
            kwargs = {
                "model": OPENAI_MODEL,
                "instructions": SYSTEM_INSTRUCTIONS,
                "input": user_input,
                "max_output_tokens": 600,
            }
            if previous_response_id is not None:
                kwargs["previous_response_id"] = previous_response_id

            response = OPENAI_CLIENT.responses.create(**kwargs)
            assistant_reply = response.output_text or ""
            previous_response_id = response.id

            print(f"\nAssistant: {assistant_reply}\n")

        except Exception as exc:
            print(f"\nError: {friendly_error(exc)}\n")


def run_openrouter_chat():
    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTIONS,
        }
    ]

    print("OPENAI_API_KEY was not visible to Python, so chat.py is falling back to OpenRouter.")
    print(f"Chat started with OpenRouter model {OPENROUTER_ROUTE_MODEL}. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input,
        })

        try:
            response = OPENROUTER_CLIENT.chat.completions.create(
                model=OPENROUTER_ROUTE_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=600,
            )

            assistant_reply = response.choices[0].message.content or ""

            print(f"\nAssistant: {assistant_reply}\n")

            messages.append({
                "role": "assistant",
                "content": assistant_reply,
            })

        except Exception as exc:
            messages.pop()
            print(f"\nError: {friendly_error(exc)}\n")


def run_chat():
    provider = active_provider()

    if provider == "openai":
        run_openai_chat()
        return

    if provider == "openrouter":
        run_openrouter_chat()
        return

    print(
        "No API key found. Set OPENAI_API_KEY for the direct OpenAI API, "
        "or OPENROUTER_API_KEY for OpenRouter."
    )


if __name__ == "__main__":
    run_chat()
