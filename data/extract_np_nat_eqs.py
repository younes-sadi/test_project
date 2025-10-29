import os, json, re
from typing import List, Dict

SRC = "artifacts/np_clean.jsonl"
OUT = "artifacts/nat_eq_candidates.jsonl"
os.makedirs("artifacts", exist_ok=True)

# allow only these tokens in Nat expressions
ALLOWED = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+*()= -"))
# must contain + or * and an '='
EQ_RE = re.compile(r"([A-Za-z0-9+\s\*\(\)]+)=([A-Za-z0-9+\s\*\(\)]+)")

def is_nat_expr(s: str) -> bool:
    # quick filter: only allowed ASCII tokens
    return all((c in ALLOWED) for c in s)

def normalize_expr(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # remove trailing/leading parentheses if redundant
    s = s.strip()
    return s

def find_eqs_in_text(s: str) -> List[str]:
    out = []
    for m in EQ_RE.finditer(s):
        lhs, rhs = m.group(1), m.group(2)
        expr = f"{lhs} = {rhs}"
        expr = normalize_expr(expr)
        if is_nat_expr(expr) and (("+" in expr) or ("*" in expr)):
            out.append(expr)
    return out

def scan_record(rec: Dict) -> List[Dict]:
    out = []
    # search in cleaned plain text
    out += [{"source": "text", "expr": e} for e in find_eqs_in_text(rec.get("text_clean",""))]
    # search in extracted formulas (LaTeX cleaned lightly)
    for f in rec.get("formulas", []):
        # replace common LaTeX for +,* and parentheses are already simple
        f2 = f.replace("\\cdot", "*").replace("\\times", "*")
        # remove ^{...} exponents (skip if present)
        f2 = re.sub(r"\^\{[^}]+\}", "", f2)
        f2 = re.sub(r"\^[0-9]+", "", f2)
        out += [{"source": "formula", "expr": e} for e in find_eqs_in_text(f2)]
    return out

def main():
    seen = set()
    kept = 0
    with open(SRC, "r", encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            candidates = scan_record(rec)
            for c in candidates:
                expr = c["expr"]
                # deduplicate exact expr per split
                key = (rec["split"], expr)
                if key in seen:
                    continue
                seen.add(key)
                out = {
                    "split": rec["split"],
                    "id": rec["id"],
                    "title": rec.get("title_clean",""),
                    "expr": expr,
                    "source": c["source"]
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1
    print(f"Wrote {kept} candidates to {OUT}")

if __name__ == "__main__":
    main()
