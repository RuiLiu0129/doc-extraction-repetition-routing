"""Check OpenRouter API key works, and print one-call cost per model.

Usage:
    pip install -r requirements.txt
    python src/check_keys.py

Needs OPENROUTER_API_KEY in a .env file in the repo root.
Model names must be verified at run time on openrouter.ai/models;
update the list below if a name errors.
"""
import os

from dotenv import load_dotenv

load_dotenv()

MODELS = [
    "openrouter/openai/gpt-5-mini",
    "openrouter/anthropic/claude-haiku-4.5",
    "openrouter/google/gemini-2.5-flash",
    "openrouter/qwen/qwen3-235b-a22b-2507",
    "openrouter/deepseek/deepseek-chat",
]

PROMPT = (
    "Question: What is 17 * 23?\n"
    "Give your answer, then on a new line write 'Confidence: X' where X is "
    "the probability (0-100) that your answer is correct."
)


def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        print("FAIL: OPENROUTER_API_KEY not set. Create a .env file (see step1-setup-v2.md).")
        raise SystemExit(1)
    import litellm

    for m in MODELS:
        try:
            r = litellm.completion(
                model=m,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=200,
            )
            text = r.choices[0].message.content.strip().replace("\n", " | ")
            cost = litellm.completion_cost(completion_response=r)
            print(f"OK   {m:40s} ${cost:.5f}  {text[:80]}")
        except Exception as e:  # noqa: BLE001 - report per-model, do not stop the check
            print(f"FAIL {m:40s} {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
