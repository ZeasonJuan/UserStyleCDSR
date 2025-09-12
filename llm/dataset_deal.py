from typing import Dict, List
import argparse

def load_item2index(path: str) -> Dict[str, int]:
    """raw_id -> new_id"""
    m = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            raw, idx = line.rstrip('\n').split('\t')
            m[raw] = int(idx)
    return m

def invert_item2index(raw2new: Dict[str, int]) -> Dict[int, str]:
    """new_id -> raw_id"""
    return {new: raw for raw, new in raw2new.items()}

def load_text(path: str) -> Dict[str, str]:
    """raw_id -> text; 跳过首行表头"""
    m = {}
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()  # 'item_id:token\ttext:token_seq\n'
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            if not parts: 
                continue
            raw = parts[0]
            text = parts[1] if len(parts) > 1 else '.'
            m[raw] = text
    return m

def inter_has_header(first_line: str) -> bool:
    """Return True if the first line looks like a UniSRec .inter header."""
    return 'item_id_list:token_seq' in first_line

def convert_inter_to_seq_text(
    inter_path: str,
    item2index_path: str,
    text_path: str,
    out_path: str,
    swap: bool = False,
    joiner: str = " "
) -> None:
    """
    Read a .inter file (new_id sequences), map to raw_id -> text, and write
    "<seq_as_new_ids>\t<concatenated_texts>" per line to out_path.
    """
    # Reuse loaders
    raw2new = load_item2index(item2index_path)
    # invert to new_id -> raw_id (keys become int by construction of load_item2index)
    new2raw = invert_item2index(raw2new)
    raw2text = load_text(text_path)

    total, wrote, missing = 0, 0, 0

    with open(inter_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        first = fin.readline()
        if not first:
            return

        seq_col = 1  # default to second column
        pending_lines = []

        if inter_has_header(first):
            # Derive column index from header just in case
            headers = first.rstrip("\n").split("\t")
            if "item_id_list:token_seq" in headers:
                seq_col = headers.index("item_id_list:token_seq")
        else:
            # First line is actual data
            pending_lines.append(first)

        # Process all lines including the first data line (if any)
        for line in pending_lines + fin.readlines():
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            cols = line.split("\t")
            if len(cols) <= seq_col:
                print(f"嘿嘿，跳过格式错误行: {line}")
                missing += 1
                continue

            seq_str = cols[seq_col]
            id_tokens = [tok for tok in seq_str.split(" ") if tok != ""]

            texts = []
            any_missing = False
            for tok in id_tokens:
                # new_id could be numeric
                try:
                    nid = int(tok)
                except ValueError:
                    nid = tok  # if your mapping uses str keys (unlikely for UniSRec)
                raw_id = new2raw.get(nid)
                if raw_id is None:
                    any_missing = True
                    continue
                txt = raw2text.get(raw_id)
                if txt is None:
                    any_missing = True
                    continue
                texts.append(txt)

            combined = joiner.join(texts)
            fout.write(f"{seq_str}\t{combined}\n")
            wrote += 1
            if any_missing:
                missing += 1

    print(f"[seq2text] lines={total}, wrote={wrote}, with_missing={missing}, out='{out_path}'")

if __name__ == "__main__":
    datasets = "office-arts"
    dataset_abb = "OA"
    dataset_all = "Office"
    train_or_test = "train"  # or "test"
    convert_inter_to_seq_text(
        inter_path=f"../dataset/{datasets}/{dataset_abb}/{dataset_abb}.{train_or_test}.inter",
        item2index_path=f"../dataset/{datasets}/{dataset_abb}/{dataset_all}.item2index",
        text_path=f"../dataset/{datasets}/{dataset_abb}/{dataset_all}.text",
        out_path=f"../dataset/{datasets}/{dataset_abb}/{dataset_abb}.seq2text_{train_or_test}.tsv",
        joiner=" "
    )