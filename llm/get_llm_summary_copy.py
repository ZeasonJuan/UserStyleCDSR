import ollama, json, os, time

def build_messages(item_tuples):
    items_block = "\n".join(f"{t}: {d}" for t, d in item_tuples)
    print("Items block:")
    print(items_block)
    return [
        {"role": "system", "content": """You are an expert in recommendation systems.
Given the texts of items a user interacted with, summarize the user's overall preference style.

INSTRUCTIONS:
1) Focus on stable long-term interests, but also note recent short-term shifts if any.
2) Avoid copying item titles verbatim; generalize to concepts/attributes.
3) Prefer signals that repeat across multiple items; discount one-offs and popularity bias.
4) Keep it concise and structured.

OUTPUT:
Return a single valid JSON object with keys:
long_term_interests, recent_trends, disliked_or_avoided, style_keywords, rationale.
No extra commentary or markdown."""},
        {"role": "user", "content": f"INPUT (each line is one item; most recent last):\n{items_block}\n"}
    ]
print("Building messages...")
res = ollama.chat(
    model="phi3.5",
    messages=build_messages([
        ("Noise-Cancelling Headphones", "over-ear; ANC; electronics; brand XYZ; travel"),
        ("Sci-Fi Novel 'Starlight Drift'", "science fiction; space; author A.B."),
    ]),
    options={"num_ctx": 32768, "num_predict": 1024},
    stream=False
)

answer = res["message"]["content"]
print(answer)
# 解析与保存
exit(0)
try:
    parsed = json.loads(answer)
except json.JSONDecodeError:
    parsed = {"raw": answer, "parse_error": True}
os.makedirs("logs", exist_ok=True)
with open("logs/user_style.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"summary": parsed}, ensure_ascii=False) + "\n")