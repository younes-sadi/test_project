import re, json, os
from datasets import load_from_disk

SRC = "data/naturalproofs_gen_full"      
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

#  utilities 

def _strip_links(m):
    # [[A|B]] -> B ; [[A]] -> A
    inner = m.group(1)
    return inner.split("|")[-1] if "|" in inner else inner

def clean_text(t: str) -> str:
    if not t:
        return ""
    # remove wiki links [[...]]
    t = re.sub(r"\[\[([^\]]+)\]\]", _strip_links, t)
    # remove templates {{...}} (non-nested best effort)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    # remove HTML tags
    t = re.sub(r"<[^>]+>", " ", t)
    # normalize LaTeX line breaks and excessive backslashes
    t = t.replace("\\\\", " ").replace("\n", " ")
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

FORMULA_PATTERNS = [
    r"\$(.+?)\$",          # $ ... $
    r"\\\((.+?)\\\)",      # \( ... \)
    r"\\\[(.+?)\\\]"       # \[ ... \]
]

def extract_formulas(raw: str):
    if not raw:
        return []
    found = []
    for pat in FORMULA_PATTERNS:
        for m in re.finditer(pat, raw):
            s = m.group(1)
            # light cleanup inside formula
            s = s.replace("\\,", " ").replace("\\;", " ").strip()
            s = re.sub(r"\s+", " ", s)
            found.append(s)
    return found

#  main 

def main():
    ds = load_from_disk(SRC)  # DatasetDict with train/validation/test
    stats = {}
    out_path = os.path.join(OUT_DIR, "np_clean.jsonl")
    with open(out_path, "w", encoding="utf-8") as fout:
        for split in ["train", "validation", "test"]:
            cnt = 0
            for ex in ds[split]:
                title_raw = ex.get("title") or ""
                text_raw  = ex.get("text")  or ""
                rec = {
                    "split": split,
                    "id": ex.get("id"),
                    "title_raw": title_raw,
                    "title_clean": clean_text(title_raw),
                    "text_clean": clean_text(text_raw),
                    "formulas": extract_formulas(text_raw)
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt += 1
            stats[split] = cnt
    with open(os.path.join(OUT_DIR, "np_clean_stats.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    print("Wrote:", out_path)
    print("Stats:", stats)

if __name__ == "__main__":
    main()
