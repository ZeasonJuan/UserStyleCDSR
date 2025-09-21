import asyncio, json, time
from pathlib import Path
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import statistics
import matplotlib.pyplot as plt
def paint(data: list): 

    x = range(len(data))  # 横坐标：索引
    y = data              # 纵坐标：数值

    plt.scatter(x, y)     # 画散点
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Scatter plot of list")
    plt.show()

BASE_URL = "http://127.0.0.1:8011/v1" #mark😁
API_KEY  = "dummy"
MODEL_ID = "phi3.5-mini"

CONCURRENCY = 64             # 并发度（按显存+模型大小调）
OUTPUT_JSONL = "llm_summary_500.jsonl"


PROMPT_TEMPLATE = """You are an expert in recommendation systems. 
Given the texts of items a user interacted with, summarize the user's overall preference style.

INPUT (the text below is the concatenation of texts of items the user interacted with, most recent last):
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
HF_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"  # 用于本地计数的 HF 模型名
MODEL_CTX_LEN = 10000                              # vLLM 启动时的 --max-model-len
MAX_NEW_TOKENS = 256                              # 你在请求里设的 max_tokens
SAFETY_MARGIN = 32  

ct_len = []
PROMPT_BUDGET = MODEL_CTX_LEN - MAX_NEW_TOKENS - SAFETY_MARGIN

_tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_ID,
    use_fast=True,
    local_files_only=True  # 没网环境读本地缓存
)

def _count_chat_tokens(messages):
    """
    计算 chat 消息在当前模型下的 token 数。
    优先使用 chat_template（与 vLLM 对齐），若不可用则退化到拼接文本。
    """
    # try:
    ids = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # 与 chat.completions 一致
    )
    return len(ids)
    # except Exception:
    #     # 退化：简单拼接角色与内容（不完全精确，但可作为兜底）
    #     concat = []
    #     for m in messages:
    #         role = m.get("role", "user")
    #         content = m.get("content", "")
    #         concat.append(f"<{role}>\n{content}\n</{role}>\n")
    #     toks = _tokenizer("".join(concat), add_special_tokens=False).input_ids
    #     return len(toks)


async def one_call(seq_items, idx, sem):
    lines = "\n".join(seq_items)
    prompt = PROMPT_TEMPLATE.format(lines=lines)
    # 原来的 messages
    messages = [
        {"role": "system", "content": "Return only the single summary paragraph which is less than 512 tokens."},
        {"role": "user", "content": prompt},
    ]

    # # 计算并截断到预算（PROMPT_BUDGET）
    # messages, trunc_info = _truncate_user_to_budget(messages, PROMPT_BUDGET)
    ct = 66666 
    loop_number = 0
    while ct >= 511:
        async with sem:
            t0 = time.perf_counter()
            resp = await client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=0.0,
                seed=42,
                max_tokens=512,
            )
            t1 = time.perf_counter()

        content = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) if usage else 0
        
        ct = getattr(usage, "completion_tokens", 0) if usage else 0
        print(f"Prompt tokens: {pt}, Completion tokens: {ct}")
        loop_number += 1
        if loop_number >= 5:
            exit(1)
        if ct >= 511:
            print(content)
            print(messages)
    ct_len.append(ct)

    return {
        "index": idx,
        "latency_sec": round(t1 - t0, 4),
        "completion_tokens": ct,
        "prompt_tokens": pt,
        "output": content,
    }


INPUT_SEQ2TEXT_TSV = "../dataset/office-arts/OA/Office.seq2text_train.tsv"       # 你的 (item_newid_seq -> items_text) 输入文件
OUTPUT_SEQ2LLM_TSV = "../dataset/office-arts/OA/Office.seq2summary_train.tsv"    # 产出的 (item_newid_seq -> llm_summary) 文件
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
        # 如果以后把 joiner 换成 " ||| "，可以改成：
        seq_items = [s for s in items_text.split("|||") if s.strip()]
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
    paint(ct_len)
    total_pt = sum(r.get("prompt_tokens", 0) for r in results)
    total_ct = sum(r.get("completion_tokens", 0) for r in results)
    total = len(results)
    #接下来计算最长输入token和最长输出token
    max_pt = max(r.get("prompt_tokens", 0) for r in results)
    max_ct = max(r.get("completion_tokens", 0) for r in results)
    #接下来计算平均token和中位数token
    mean_pt = statistics.mean(r.get("prompt_tokens", 0) for r in results)
    mean_ct = statistics.mean(r.get("completion_tokens", 0) for r in results)
    median_pt = statistics.median(r.get("prompt_tokens", 0) for r in results)
    median_ct = statistics.median(r.get("completion_tokens", 0) for r in results)
    overtoken = sum(1 for r in results if r.get("completion_tokens", 0) >= 512)
    print(f"Max tokens prompt/completion: {max_pt}/{max_ct}")
    print(f"Mean tokens prompt/completion: {mean_pt:.2f}/{mean_ct:.2f}")
    print(f"Median tokens prompt/completion: {median_pt}/{median_ct}")
    print(f"Over token limit: {overtoken}, over rate: {overtoken/total:.2%}")

    dur = t1 - t0
    print(f"Wrote {total} lines to {OUTPUT_SEQ2LLM_TSV}")
    print(f"Done {total} requests in {dur:.2f}s | {total/dur:.2f} req/s")
    print(f"Tokens prompt/completion: {total_pt}/{total_ct} | total {total_pt+total_ct}")

if __name__ == "__main__":
    asyncio.run(main())