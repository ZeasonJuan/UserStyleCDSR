# -*- coding: utf-8 -*-
import os, re, math, numpy as np, torch
from transformers import AutoTokenizer, AutoModel, logging
from numpy.lib.format import open_memmap
from tqdm import tqdm

# ----------------- 配置 -----------------
emb_type = "CLS"                      # "CLS" 或 "Mean"
plm_name = "bert-base-uncased"
device = "cuda:4"                     # 你的显卡
batch_size = 128                      # 按显存调
max_len = 512                         # BERT base 最大就是 512
l2_normalize = False                  # 是否对向量做 L2 归一化

INPUT_SEQ2LLM_TSV   = "../dataset/office-arts/OA/Arts.seq2summary_test.tsv"
OUTPUT_BASENAME     = "../dataset/office-arts/OA/Arts.seq2bert_test"  # 会生成 .npy 和 .map.tsv

CACHE_DIR = "/mnt/disk2/tingtao.zheng/hf_cache/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"  # 你的 transformers 缓存目录
# 要去掉的前缀（忽略大小写；只在句首才去）
PREFIX_PAT = re.compile(r"^\s*the user demonstrates a consistent\s*", re.IGNORECASE)

# ----------------- 函数 -----------------
def load_plm(model_name="bert-base-uncased"):
    logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(
        CACHE_DIR,
        local_files_only=True,   # 关键：强制只用本地缓存
    )
    mdl = AutoModel.from_pretrained(
        CACHE_DIR,
        local_files_only=True,
    )
    mdl.eval().to(device)
    return tok, mdl

def clean_prefix(text: str) -> str:
    return re.sub(PREFIX_PAT, "", text).strip()

def read_seq_and_summary(tsv_path):
    """假设每行形如：<item_id_list>\t<llm_summary>
       第一行若是表头会自动跳过。"""
    seqs, sums = [], []
    with open(tsv_path, "r", encoding="utf-8") as f:
        first = f.readline()
        # 尝试判断是否表头（包含非数字/空格/制表符且不含空格分隔的 id）
        if first.count("\t") == 0:
            raise ValueError("TSV 至少需要两列：item_id_list<TAB>llm_summary")
        c0, c1 = first.rstrip("\n").split("\t", 1)
        looks_like_header = any(k in c0.lower() for k in ["seq", "item", "list"]) or any(
            k in c1.lower() for k in ["summary", "desc"]
        )
        if not looks_like_header:
            seqs.append(c0)
            sums.append(c1)
        # 其余行
        for line in f:
            if not line.strip():
                continue
            s0, s1 = line.rstrip("\n").split("\t", 1)
            seqs.append(s0)
            sums.append(s1)
    return seqs, sums

@torch.no_grad()
def encode_summaries(seqs, summaries, tokenizer, model):
    """流式两遍：先确定 N 和 dim，再 memmap 写入，避免一次性占内存。"""
    N = len(summaries)

    # 先跑一个样本拿到 hidden dim
    tmp = tokenizer(
        summaries[0] or ".", padding=True, truncation=True,
        max_length=max_len, return_tensors="pt"
    ).to(device)
    out = model(**tmp)
    hidden = out.last_hidden_state.shape[-1]

    # 预分配 memmap
    npy_path = OUTPUT_BASENAME + ".npy"
    mmap = open_memmap(npy_path, mode="w+", dtype="float32", shape=(N, hidden)) 

    # 写出 map 文件（seq -> row_index）
    map_path = OUTPUT_BASENAME + ".map.tsv"
    with open(map_path, "w", encoding="utf-8") as mf:
        mf.write("row_index\titem_id_list\n")
        for i, seq in enumerate(seqs):
            mf.write(f"{i}\t{seq}\n")

    # 分批编码
    pbar = tqdm(range(0, N, batch_size), desc="Encoding summaries")
    for st in pbar:
        ed = min(st + batch_size, N)
        batch_texts = [clean_prefix(t) if t else "." for t in summaries[st:ed]]

        enc = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        ).to(device)
        out = model(**enc)  # last_hidden_state: [B, T, H]

        if emb_type.upper() == "CLS":
            emb = out.last_hidden_state[:, 0, :]                    # [CLS]
        else:  # Mean Pool（不含 [CLS]）
            mask = enc["attention_mask"].unsqueeze(-1).float()      # [B, T, 1]
            # 去掉 [CLS] 位置
            masked = out.last_hidden_state[:, 1:, :] * mask[:, 1:, :]
            denom = mask[:, 1:, :].sum(dim=1).clamp_min(1e-6)       # [B, 1]
            emb = masked.sum(dim=1) / denom

        if l2_normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)

        mmap[st:ed, :] = emb.detach().cpu().float().numpy()

    mmap.flush()
    return npy_path, map_path

# ----------------- 主流程 -----------------
if __name__ == "__main__":
    print("Loading PLM...")
    tokenizer, model = load_plm(plm_name)

    print(f"Reading {INPUT_SEQ2LLM_TSV} ...")
    seqs, summaries = read_seq_and_summary(INPUT_SEQ2LLM_TSV)
    print(f"Total rows: {len(seqs)}")

    npy_path, map_path = encode_summaries(seqs, summaries, tokenizer, model)
    print(f"Done.\nEmbeddings: {npy_path}\nMapping:    {map_path}")