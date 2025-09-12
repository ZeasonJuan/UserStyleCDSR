import asyncio, json, time
from pathlib import Path
from openai import AsyncOpenAI

BASE_URL = "http://127.0.0.1:8010/v1"
API_KEY  = "dummy"
MODEL_ID = "phi3.5-mini"

CONCURRENCY = 64             # 并发度（按显存+模型大小调）
MAX_INPUT_LINES = 32         # 每条序列最多送多少 item 行，防止prompt过长
OUTPUT_JSONL = "llm_summary_500.jsonl"


PROMPT_TEMPLATE = """You are an expert in recommendation systems. 
Given the texts of items a user interacted with, summarize the user’s overall preference style.

INPUT (each line is one item the user engaged with; most recent first):
{lines}

INSTRUCTIONS:
1) Focus on stable long-term interests, but also note recent short-term shifts if any.
2) Avoid copying item titles verbatim; generalize to concepts/attributes.
3) Prefer signals that repeat across multiple items; discount one-offs and popularity bias.
4) Keep it concise and structured.

OUTPUT:
A single paragraph summarizing the user's preference style, just provide the summary; no other wording is needed..
"""

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

async def one_call(seq_items, idx, sem):
    lines = "\n".join(seq_items[:MAX_INPUT_LINES])
    prompt = PROMPT_TEMPLATE.format(lines=lines)

    async with sem:
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role":"system","content":"Return only the summary paragraph."},
                {"role":"user","content": prompt}
            ],
            temperature=0.0,  # 便于验证“stateless + 可复现”
            seed=42,          # vLLM 支持 per-request seed
            max_tokens=256
        )
        t1 = time.perf_counter()

    content = resp.choices[0].message.content
    usage = getattr(resp, "usage", None) 
    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
    ct = getattr(usage, "completion_tokens", 0) if usage else 0

    return {
        "index": idx,
        "latency_sec": round(t1 - t0, 4),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "output": content
    }


INPUT_SEQ2TEXT_TSV = "../dataset/or-pantry/OP/Pantry.seq2text.tsv"       # 你的 (item_newid_seq -> items_text) 输入文件
OUTPUT_SEQ2LLM_TSV = "../dataset/or-pantry/OP/Pantry.seq2summary.tsv"    # 产出的 (item_newid_seq -> llm_summary) 文件
JOINER = " "                                      # 你当前的 joiner

# ====== 小工具：读取 seq2text TSV ======
def load_seq2text_tsv(path):
    """
    读取两列 TSV：<seq_str>\t<items_text>
    返回列表 [(seq_str, items_text), ...]
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            seq_str, items_text = parts
            out.append((seq_str, items_text))
    return out

async def main():
    # 读取 500 条
    pairs = load_seq2text_tsv(INPUT_SEQ2TEXT_TSV)  # [(seq_str, items_text), ...]
    n = len(pairs)
    print(f"Loaded {n} sequences from {INPUT_SEQ2TEXT_TSV}")

    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.perf_counter()
    tasks = []

    for i, (seq_str, items_text) in enumerate(pairs):
        # 你的 joiner 现在是 " "，无法按 item 还原，只能整段作为一行送入
        seq_items = [items_text]
        # 如果以后把 joiner 换成 " ||| "，可以改成：
        # seq_items = [s for s in items_text.split(" ||| ") if s.strip()]
        tasks.append(asyncio.create_task(one_call(seq_items, i, sem)))

    results = await asyncio.gather(*tasks)
    t1 = time.perf_counter()

    # 写出
    with open(OUTPUT_SEQ2LLM_TSV, "w", encoding="utf-8") as fout:
        for r in results:
            idx = r["index"]
            seq_str = pairs[idx][0]
            summary = (r["output"] or "").strip().replace("\n", " ")
            fout.write(f"{seq_str}\t{summary}\n")

    # 统计
    total_pt = sum(r.get("prompt_tokens", 0) for r in results)
    total_ct = sum(r.get("completion_tokens", 0) for r in results)
    total = len(results)
    dur = t1 - t0
    print(f"Wrote {total} lines to {OUTPUT_SEQ2LLM_TSV}")
    print(f"Done {total} requests in {dur:.2f}s | {total/dur:.2f} req/s")
    print(f"Tokens prompt/completion: {total_pt}/{total_ct} | total {total_pt+total_ct}")

if __name__ == "__main__":
    asyncio.run(main())