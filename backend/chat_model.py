# chat_model.py
# Groq wrapper used by the final answer generation.
# Keeps the model interface simple, similar to the sample EuriChatModel.

import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
_CHAT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

_total_tokens = 0


def reset_token_count():
    global _total_tokens
    _total_tokens = 0


def get_token_count() -> int:
    return _total_tokens


class GroqChatModel:
    # Single Groq client wrapper so generation always uses the configured model.
    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
    ) -> str:

        used_model = model or _CHAT_MODEL
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            start = time.perf_counter()

            try:
                response = _client.chat.completions.create(
                    model=used_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                break

            except Exception:
                if attempt == max_attempts:
                    raise

                wait = 2 * attempt
                print(
                    f"[{used_model}] Groq error, retrying in "
                    f"{wait}s (attempt {attempt}/{max_attempts})"
                )
                time.sleep(wait)

        elapsed = time.perf_counter() - start

        tokens = response.usage.total_tokens if response.usage else "n/a"

        if isinstance(tokens, int):
            global _total_tokens
            _total_tokens += tokens

        print(f"[{used_model}] {tokens} tokens, {elapsed:.1f}s")

        return response.choices[0].message.content