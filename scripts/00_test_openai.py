# scripts/00_test_openai.py
from grpo_reasoning.utils.env import require_env
from openai import OpenAI

client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
resp = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
    max_tokens=10,
)
print(resp.choices[0].message.content)
print(f"Tokens used: {resp.usage.total_tokens}")