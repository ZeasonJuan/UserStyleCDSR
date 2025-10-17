import asyncio, json, time
from pathlib import Path
from openai import AsyncOpenAI
from transformers import AutoTokenizer

BASE_URL = "http://127.0.0.1:8011/v1" #mark😁
API_KEY  = "dummy"
MODEL_ID = "phi3.5-mini"
INPUT_SEQ2TEXT_TSV = "../dataset/ot-pantry/OP/OR.seq2text_test.tsv"  
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
HF_MODEL_ID = "microsoft/Phi-3.5-mini-instruct" 

_tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_ID,
    use_fast=True,
    local_files_only=True  # 没网环境读本地缓存
)

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

messages = [
        {"role": "system", "content": "Return only the summary paragraph."},
        {"role": "user", "content": prompt},
    ]


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
            len
            out.append((seq_str, items_text))
    return out