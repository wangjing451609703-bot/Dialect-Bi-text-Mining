"""
Replace bar-word to de and generate German-like documents.
Use DiaLemma to enable word-by-word substitution and build qrel map to match query-document pairs.
"""

import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple


def read_tsv_pairs(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            rows.append((parts[0], parts[1]))
    return rows

def write_tsv_pairs(path: Path, rows: List[Tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k, v in rows:
            f.write(f"{k}\t{v}\n")

def build_qrels_map(qrels_path: Path) -> Dict[str, str]:
    mp: Dict[str,str] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            qid, docid, rel = parts[0], parts[1], parts[2]
            if qid not in mp:
                mp[qid] = docid
    return mp


def load_dict_from_dialemma(jsonl_path: Path, lowercase: bool) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                line = line.rstrip(",")
                obj = json.loads(line)
            de = obj.get("term")
            if not isinstance(de, str) or not de.strip():
                continue
            cands = []
            dts = obj.get("translations") or []
            if isinstance(dts, list):
                for dt in dts:
                    if isinstance(dt, str) and dt.strip():
                        cands.append(dt.strip())
            vs = obj.get("inflected_variants") or []
            if isinstance(vs, list):
                for v in vs:
                    if isinstance(v, str) and v.strip():
                        cands.append(v.strip())
            for token in cands:
                key = token.strip()
                if lowercase:
                    key = key.lower()
                if key and key not in mp:
                    mp[key] = de.strip()
    return mp

def replace_sentence_tokens(sent: str, repl_map: Dict[str,str], lowercase: bool) -> Tuple[str, int]:
    toks = sent.strip().split()
    rep = 0
    out = []
    for t in toks:
        k = t.lower() if lowercase else t
        if k in repl_map:
            out.append(repl_map[k])
            rep += 1
        else:
            out.append(t)
    return (" ".join(out), rep)

def process_split(step1_split_dir: Path, data_split_dir: Path, out_split_dir: Path,
                  dialemma_path: Path, 
                  lowercase_match: bool, keep_if_replaced_only: bool):
    qrels_path = data_split_dir / "qrels.tsv"
    q2d = build_qrels_map(qrels_path)

    dicts = load_dict_from_dialemma(dialemma_path, lowercase_match)

    bar_q_path = step1_split_dir / "bar_queries.tsv"

    rows = read_tsv_pairs(bar_q_path)

    out_rows: List[Tuple[str,str]] = []
    miss_qrels = 0
    skipped = 0
    replaced_sents = 0
    replaced_tokens = 0

    for qid, qtext in rows:
        de_text, nrep = replace_sentence_tokens(qtext, dicts, lowercase_match)
        if keep_if_replaced_only and nrep <= 0:
            skipped += 1
            continue

        docid = q2d.get(qid)
        if not docid:
            miss_qrels += 1
            continue

        out_rows.append((docid, de_text))
        if nrep > 0:
            replaced_sents += 1
            replaced_tokens += nrep

    out_path = out_split_dir / f"bar_dict_trans_corpus.tsv"
    write_tsv_pairs(out_path, out_rows)
    print(f"[{step1_split_dir.name}] bar2de: wrote {len(out_rows)} rows; "
            f"replaced_sents={replaced_sents}; replaced_tokens={replaced_tokens}; "
            f"skipped(no replace)={skipped}; missing_qrels={miss_qrels}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--dialemma_path", required=True)
    ap.add_argument("--lowercase_match", action="store_true", help="Lowercase keys and tokens for exact match")
    ap.add_argument("--keep_if_replaced_only", action="store_true", help="Keep a sentence only if >=1 token replaced")
    args = ap.parse_args()

    step1_root = Path(args.step1_root)
    data_root  = Path(args.data_root)
    out_root   = Path(args.out_root)
    dialemma_path  = Path(args.dialemma_path)


    splits = []
    for name in ("train","dev"):
        s1 = step1_root / name
        d0 = data_root / name
        if s1.exists() and d0.exists():
            splits.append((s1, d0, out_root / name))
    if not splits:
        print("[ERR] no valid splits found.")
        return

    for s1_dir, d0_dir, out_dir in splits:
        out_dir.mkdir(parents=True, exist_ok=True)
        process_split(s1_dir, d0_dir, out_dir, dialemma_path,
                      args.lowercase_match, args.keep_if_replaced_only)

    print(f"[OK] done. Outputs: {out_root}/<split>/bar_dict_trans_corpus.tsv")

if __name__ == "__main__":
    main()
