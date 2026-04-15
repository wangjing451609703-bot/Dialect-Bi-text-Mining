"""
Performs the final sampling split. Select fixed numbers for train and dev at the desired size.
"""

import argparse, random
from pathlib import Path
from typing import Dict, List, Tuple

def read_tsv_pairs(path: Path) -> Dict[str, str]:
    mp = {}
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln: continue
            pid, text = ln.split("\t", 1)
            if pid not in mp:
                mp[pid] = text
    return mp

def read_qrels(path: Path) -> List[Tuple[str,str,int]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = ln.split("\t")
            qid, did = parts[0], parts[1]
            rows.append((qid, did))
    return rows

def write_tsv_pairs(path: Path, rows: List[Tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a, b in rows:
            f.write(f"{a}\t{b}\n")

def write_qrels(path: Path, rows: List[Tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a, b in rows:
            f.write(f"{a}\t{b}\t1\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True,  help="Folder containing merged queries.tsv, corpus.tsv, qrels.tsv")
    ap.add_argument("--out_root", required=True, help="Output root; will create train/ and dev/")
    ap.add_argument("--n_train", type=int, default=32458)
    ap.add_argument("--n_dev",   type=int, default=3605)
    ap.add_argument("--seed",    type=int, default=2025)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    q_path = in_root / "queries.tsv"
    d_path = in_root / "corpus.tsv"
    r_path = in_root / "qrels.tsv"
    assert q_path.exists() and d_path.exists() and r_path.exists(), "Missing merged queries/corpus/qrels."

    qid2txt = read_tsv_pairs(q_path)
    did2txt = read_tsv_pairs(d_path)
    qrels   = read_qrels(r_path)


    cand = []
    seen = set()
    for qid, did in qrels:
        if (qid in qid2txt) and (did in did2txt):
            key = (qid, did)
            if key not in seen:
                seen.add(key)
                cand.append(key)

    if len(cand) < (args.n_train + args.n_dev):
        raise RuntimeError(f"Not enough positive pairs: have {len(cand)}, need {args.n_train + args.n_dev}.")

    rnd = random.Random(args.seed)
    rnd.shuffle(cand)

    train_pairs = cand[:args.n_train]
    dev_pairs   = cand[args.n_train:args.n_train+args.n_dev]

    def build_split(pairs: List[Tuple[str,str]], split_dir: Path):
        q_used = sorted({q for q, _ in pairs})
        d_used = sorted({d for _, d in pairs})
        queries_out = [(q, qid2txt[q]) for q in q_used]
        corpus_out  = [(d, did2txt[d]) for d in d_used]
        qrels_out   = [(q, d) for q, d in pairs]
        split_dir.mkdir(parents=True, exist_ok=True)
        write_tsv_pairs(split_dir / "queries.tsv", queries_out)
        write_tsv_pairs(split_dir / "corpus.tsv",  corpus_out)
        write_qrels(split_dir / "qrels.tsv",       qrels_out)

    build_split(train_pairs, out_root / "train")
    build_split(dev_pairs,   out_root / "dev")

    print(f"[OK] train={len(train_pairs)} dev={len(dev_pairs)}; total candidates={len(cand)}; seed={args.seed}")

if __name__ == "__main__":
    main()
