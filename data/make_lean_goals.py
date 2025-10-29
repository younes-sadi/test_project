import os, json, re, random, math
from collections import defaultdict

SRC = "artifacts/nat_eq_candidates.jsonl"
OUT_DIR = "data"
SEED = 42
random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

VAR_RE = re.compile(r"[A-Za-z]")

def infer_vars(expr: str):
    # collect variable letters in order of first appearance
    seen, order = set(), []
    for ch in expr:
        if VAR_RE.fullmatch(ch) and ch not in seen:
            seen.add(ch); order.append(ch)
    # restrict to 1..4 variables for simple Nat goals
    return order[:4]

def sanitize_goal(expr: str):
    # collapse spaces around operators
    expr = re.sub(r"\s*\+\s*", " + ", expr)
    expr = re.sub(r"\s*\*\s*", " * ", expr)
    expr = re.sub(r"\s*=\s*", " = ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr

def mk_goal_name(prefix: str, idx: int):
    return f"{prefix}_{idx:06d}"

def rec_to_goal_obj(rec, idx):
    expr = sanitize_goal(rec["expr"])
    vars_ = infer_vars(expr)
    if len(vars_) == 0:
        vars_ = ["n"]
    params = " ".join(vars_)
    params_str = f"({params} : Nat)" if len(vars_)>0 else ""
    name = mk_goal_name("nat_eq", idx)
    return {
        "name": name,
        "params": params_str,
        "goal": expr,
        "category": "NatEq",
        "source_split": rec["split"],
        "source_id": rec.get("id"),
        "title": rec.get("title","")
    }

def main():
    records = []
    with open(SRC, "r", encoding="utf-8") as fin:
        for line in fin:
            records.append(json.loads(line))

    # group by original split to preserve distribution
    by_split = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)

    goals_all = []
    idx = 0
    for split, items in by_split.items():
        # shuffle deterministically
        random.Random(SEED).shuffle(items)
        for r in items:
            g = rec_to_goal_obj(r, idx)
            goals_all.append(g)
            idx += 1

    # produce deterministic 80/10/10 over the concatenated list
    N = len(goals_all)
    n_train = math.floor(0.8 * N)
    n_val   = math.floor(0.1 * N)
    train = goals_all[:n_train]
    val   = goals_all[n_train:n_train+n_val]
    test  = goals_all[n_train+n_val:]

    def write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    write_json(os.path.join(OUT_DIR, "goals_train.json"), train)
    write_json(os.path.join(OUT_DIR, "goals_val.json"),   val)
    write_json(os.path.join(OUT_DIR, "goals_test.json"),  test)

    print("Wrote:",
          os.path.join(OUT_DIR, "goals_train.json"), len(train),
          "|", os.path.join(OUT_DIR, "goals_val.json"), len(val),
          "|", os.path.join(OUT_DIR, "goals_test.json"), len(test))

if __name__ == "__main__":
    main()
