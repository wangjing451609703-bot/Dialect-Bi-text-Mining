"""
Match the newly translated document with its original corresponding query. Create parallel query-document pairs.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

DIALECTS = ("nds", "bar", "gsw")

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
        for a, b in rows:
            f.write(f"{a}\t{b}\n")

def build_doc_to_qids_map(qrels_path: Path) -> Dict[str, List[str]]:
    mp: Dict[str, List[str]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            qid, docid, rel = parts[0], parts[1], parts[2]
            mp.setdefault(docid, []).append(qid)
    return mp

def build_qid_to_qtext_map(queries_path: Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            qid, qtext = parts[0], parts[1]
            if qid not in mp:
                mp[qid] = qtext
    return mp

def process_split(step2_split_dir: Path, data_split_dir: Path, out_split_dir: Path,
                  qid_prefix: str, doc_prefix: str, pad: int):
    qrels_path = data_split_dir / "qrels.tsv"
    queries_path = data_split_dir / "queries.tsv"
    if not qrels_path.exists() or not queries_path.exists():
        print(f"[SKIP] missing queries/qrels in {data_split_dir}")
        return

    doc2qids = build_doc_to_qids_map(qrels_path)
    qid2qtext = build_qid_to_qtext_map(queries_path)

    aligned: List[Tuple[str, str]] = []
    stats = {"total_rows": 0, "total_pairs": 0, "missing_qids": 0, "missing_qtext": 0}

    for d in ("nds","bar","gsw"):
        fpath = step2_split_dir / f"{d}_dict_trans_corpus.tsv"
        if not fpath.exists():
            print(f"[{data_split_dir.name}] warn: missing {fpath.name}, skip")
            continue
        rows = read_tsv_pairs(fpath)
        stats["total_rows"] += len(rows)
        for docid, de_text in rows:
            qids = doc2qids.get(docid, [])
            if not qids:
                stats["missing_qids"] += 1
                continue
            for qid in qids:
                qtext = qid2qtext.get(qid)
                if qtext is None:
                    stats["missing_qtext"] += 1
                    continue
                aligned.append((qtext, de_text))
                stats["total_pairs"] += 1

    queries_out: List[Tuple[str, str]] = []
    corpus_out:  List[Tuple[str, str]] = []
    qrels_out:   List[Tuple[str, str]] = []

    for idx, (qtext, de_text) in enumerate(aligned, start=1):
        sec_qid = f"{qid_prefix}{idx:0{pad}d}"
        sec_doc = f"{doc_prefix}{idx:0{pad}d}"
        queries_out.append((sec_qid, qtext))
        corpus_out.append((sec_doc, de_text))
        qrels_out.append((sec_qid, sec_doc))

    out_split_dir.mkdir(parents=True, exist_ok=True)
    write_tsv_pairs(out_split_dir / "dia2de_dict_queries.tsv", queries_out)
    write_tsv_pairs(out_split_dir / "dia2de_dict_corpus.tsv",  corpus_out)
    with (out_split_dir / "dia2de_dict_qrels.tsv").open("w", encoding="utf-8") as f:
        for sec_qid, sec_doc in qrels_out:
            f.write(f"{sec_qid}\t{sec_doc}\t1\n")

    print(f"[{data_split_dir.name}] aligned={stats['total_pairs']} from rows={stats['total_rows']} "
          f"(missing_qids={stats['missing_qids']}, missing_qtext={stats['missing_qtext']}). "
          f"Wrote: {out_split_dir}/dia2de_dict_*.tsv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2_root", required=True, help="Root of Step-2 outputs (contains <split>/<dial>_dict_trans_corpus.tsv)")
    ap.add_argument("--data_root", required=True,  help="Root of original data (contains <split>/queries.tsv and qrels.tsv)")
    ap.add_argument("--out_root",  required=True,  help="Output root for Step-4 files")
    ap.add_argument("--qid_prefix", default="SQ", help="Prefix for newly assigned sec_qid")
    ap.add_argument("--doc_prefix", default="SD", help="Prefix for newly assigned sec_docid")
    ap.add_argument("--pad", type=int, default=7, help="Zero padding width for sec ids")
    args = ap.parse_args()

    step2_root = Path(args.step2_root)
    data_root  = Path(args.data_root)
    out_root   = Path(args.out_root)

    splits = []
    for name in ("train", "dev"):
        s2 = step2_root / name
        d0 = data_root  / name
        if s2.exists() and d0.exists():
            splits.append((s2, d0, out_root / name))
    if not splits:
        print(f"[ERR] No valid split pairs under {step2_root} and {data_root}")
        return

    for s2_dir, d0_dir, out_dir in splits:
        process_split(s2_dir, d0_dir, out_dir, args.qid_prefix, args.doc_prefix, args.pad)

    print(f"[OK] Step-4 finished. Outputs: {out_root}/<split>/dia2de_dict_*.tsv")

if __name__ == "__main__":
    main()
