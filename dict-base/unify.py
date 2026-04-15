"""
Merge the newly create parallel query-document pairs into one pool. Apply deduplication.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set

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

def read_qrels(path: Path) -> List[Tuple[str, str, str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            rows.append((parts[0], parts[1]))
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

def merge_dict_keep_first(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    # Merge two id->text dicts, preferring existing entries on key collision.
    # If a key exists in both with different text, keep the first and ignore the second.
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
    return out

def unify_and_filter(step4_split_dir: Path, step5_split_dir: Path, out_split_dir: Path):
    s4_q = step4_split_dir / "dia2de_dict_queries.tsv"
    s4_c = step4_split_dir / "dia2de_dict_corpus.tsv"
    s4_r = step4_split_dir / "dia2de_dict_qrels.tsv"

    s5_q = step5_split_dir / "de2dia_dict_queries.tsv"
    s5_c = step5_split_dir / "de2dia_dict_corpus.tsv"
    s5_r = step5_split_dir / "de2dia_dict_qrels.tsv"

    for p in (s4_q, s4_c, s4_r, s5_q, s5_c, s5_r):
        if not p.exists():
            print(f"[WARN] missing file: {p}")

    queries_maps_4 = dict(read_tsv_pairs(s4_q)) if s4_q.exists() else {}
    queries_maps_5 = dict(read_tsv_pairs(s5_q)) if s5_q.exists() else {}
    corpus_maps_4  = dict(read_tsv_pairs(s4_c)) if s4_c.exists() else {}
    corpus_maps_5  = dict(read_tsv_pairs(s5_c)) if s5_c.exists() else {}

    queries_map = merge_dict_keep_first(queries_maps_4, queries_maps_5)
    corpus_map  = merge_dict_keep_first(corpus_maps_4,  corpus_maps_5)

    seen_qrels = set()
    qrels_rows: List[Tuple[str,str]] = []
    for p in (s4_r, s5_r):
        if p.exists():
            for row in read_qrels(p):
                if row not in seen_qrels:
                    seen_qrels.add(row)
                    qrels_rows.append(row)

    to_remove: Set[int] = set()
    for i, (qid, did) in enumerate(qrels_rows):
        qtext = (queries_map.get(qid) or "").strip()
        dtext = (corpus_map.get(did)  or "").strip()
        if qtext and dtext and qtext == dtext:
            to_remove.add(i)

    qrels_filtered = [row for i, row in enumerate(qrels_rows) if i not in to_remove]

    used_qids = {q for q, _ in qrels_filtered}
    used_dids = {d for _, d in qrels_filtered}

    queries_final = [(qid, queries_map[qid]) for qid in used_qids if qid in queries_map]
    corpus_final  = [(did, corpus_map[did])   for did in used_dids if did in corpus_map]

    queries_final.sort(key=lambda x: x[0])
    corpus_final.sort(key=lambda x: x[0])

    out_split_dir.mkdir(parents=True, exist_ok=True)
    write_tsv_pairs(out_split_dir / "queries.tsv", queries_final)
    write_tsv_pairs(out_split_dir / "corpus.tsv", corpus_final)
    write_qrels(out_split_dir / "qrels.tsv", qrels_filtered)

    print(f"[{out_split_dir.name}] merged qrels={len(qrels_rows)} -> kept={len(qrels_filtered)}; "
          f"removed_identicals={len(to_remove)}; queries={len(queries_final)}; corpus={len(corpus_final)}.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dia2de_root", required=True, help="Root of Step-3 outputs (contains <split>/dia2de_dict_*.tsv)")
    ap.add_argument("--de2dia_root", required=True, help="Root of Step-3 outputs (contains <split>/de2dia_dict_*.tsv)")
    ap.add_argument("--out_root",  required=True,  help="Output root for merged final files")
    args = ap.parse_args()

    step4_root = Path(args.dia2de_root)
    step5_root = Path(args.de2dia_root)
    out_root   = Path(args.out_root)

    splits = []
    for name in ("train","dev"):
        s4 = step4_root / name
        s5 = step5_root / name
        if s4.exists() and s5.exists():
            splits.append((s4, s5, out_root / name))
    if not splits:
        print(f"[ERR] No valid split pairs under {step4_root} and {step5_root}")
        return

    for s4_dir, s5_dir, out_dir in splits:
        unify_and_filter(s4_dir, s5_dir, out_dir)

    print(f"[OK] Step-6 (no reindex) finished. Outputs: {out_root}/<split>/(queries|corpus|qrels).tsv")

if __name__ == "__main__":
    main()
