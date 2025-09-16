import asyncio, json, time
from pathlib import Path
from openai import AsyncOpenAI
from transformers import AutoTokenizer

BASE_URL = "http://127.0.0.1:8010/v1" #mark😁
API_KEY  = "dummy"
MODEL_ID = "phi3.5-mini"

CONCURRENCY = 64             # 并发度（按显存+模型大小调）
MAX_INPUT_LINES = 32         # 每条序列最多送多少 item 行，防止prompt过长
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
MODEL_CTX_LEN = 4096                              # vLLM 启动时的 --max-model-len
MAX_NEW_TOKENS = 256                              # 你在请求里设的 max_tokens
SAFETY_MARGIN = 32  


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
    try:
        ids = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # 与 chat.completions 一致
        )
        return len(ids)
    except Exception:
        # 退化：简单拼接角色与内容（不完全精确，但可作为兜底）
        concat = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            concat.append(f"<{role}>\n{content}\n</{role}>\n")
        toks = _tokenizer("".join(concat), add_special_tokens=False).input_ids
        return len(toks)


def _truncate_user_to_budget(messages, budget_tokens):
    """
    若超出 budget_tokens，则只截断最后一条 user 消息的 content。
    用二分法按字符长度截断，并以空白处分词边界为优先。
    返回: (new_messages, info_dict)
    """
    if not messages:
        return messages, {"truncated": False, "tokens": 0}

    total_tokens = _count_chat_tokens(messages)
    if total_tokens <= budget_tokens:
        return messages, {"truncated": False, "tokens": total_tokens}

    # 找到最后一个 user 消息（通常就是你拼 prompt 的那条）
    idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            idx = i
            break
    if idx is None:
        # 如果没有 user，就不动
        return messages, {"truncated": False, "tokens": total_tokens}

    user_content = messages[idx].get("content", "")
    if not user_content:
        return messages, {"truncated": False, "tokens": total_tokens}

    # 二分搜索可容纳的最大字符前缀
    lo, hi = 1, len(user_content)
    best_len = 0

    def tokens_with_prefix(nchars: int) -> int:
        new_user_text = user_content[:nchars]
        # 优先切到上一个空白，避免截断在词中部
        ws = new_user_text.rfind(" ")
        if 64 < ws < nchars:  # 保证不要过早切太多
            new_user_text = new_user_text[:ws]
        tmp = list(messages)
        tmp[idx] = {**tmp[idx], "content": new_user_text}
        return _count_chat_tokens(tmp)

    # 如果连 1 个字符都放不下，就直接返回最短内容
    if tokens_with_prefix(1) > budget_tokens:
        new_msgs = list(messages)
        new_msgs[idx] = {**new_msgs[idx], "content": ""}
        return new_msgs, {"truncated": True, "tokens": _count_chat_tokens(new_msgs)}

    # 二分找最大可行前缀
    while lo <= hi:
        mid = (lo + hi) // 2
        cnt = tokens_with_prefix(mid)
        if cnt <= budget_tokens:
            best_len = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # 应用截断
    new_text = user_content[:best_len]
    ws = new_text.rfind(" ")
    if 64 < ws < best_len:
        new_text = new_text[:ws]

    new_messages = list(messages)
    new_messages[idx] = {**new_messages[idx], "content": new_text}
    final_tokens = _count_chat_tokens(new_messages)
    return new_messages, {"truncated": True, "tokens": final_tokens}

async def one_call(seq_items, idx, sem):
    lines = "\n".join(seq_items[:MAX_INPUT_LINES])
    prompt = PROMPT_TEMPLATE.format(lines=lines)

    # 原来的 messages
    messages = [
        {"role": "system", "content": "Return only the summary paragraph."},
        {"role": "user", "content": prompt},
    ]

    # 计算并截断到预算（PROMPT_BUDGET）
    messages, trunc_info = _truncate_user_to_budget(messages, PROMPT_BUDGET)

    async with sem:
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.0,
            seed=42,
            max_tokens=MAX_NEW_TOKENS,
        )
        t1 = time.perf_counter()

    content = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
    ct = getattr(usage, "completion_tokens", 0) if usage else 0

    return {
        "index": idx,
        "latency_sec": round(t1 - t0, 4),
        "prompt_tokens": pt or trunc_info.get("tokens", 0),  # vLLM会返回更准的usage
        "completion_tokens": ct,
        "truncated": trunc_info.get("truncated", False),
        "output": content,
    }


INPUT_SEQ2TEXT_TSV = "../dataset/office-arts/OA/Office.seq2text.tsv"       # 你的 (item_newid_seq -> items_text) 输入文件
OUTPUT_SEQ2LLM_TSV = "../dataset/office-arts/OA/Office.seq2summary.tsv"    # 产出的 (item_newid_seq -> llm_summary) 文件
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