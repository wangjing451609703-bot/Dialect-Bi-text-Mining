import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import re

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

def strip_variant(qid: str) -> str:
    return re.sub(r"-v\d+$", "", qid)

def build_qid_to_docids_map(qrels_path: Path) -> Dict[str, List[str]]:
    mp: Dict[str, List[str]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            qid, docid, rel = parts[0], parts[1], parts[2]
            mp.setdefault(qid, []).append(docid)
    return mp

def build_docid_to_detext_map(corpus_path: Path) -> Dict[str, str]:
    return {k: v for k, v in read_tsv_pairs(corpus_path)}

def process_split(step3_split_dir: Path, data_split_dir: Path, out_split_dir: Path,
                  qid_prefix: str, doc_prefix: str, pad: int):
    qrels_path = data_split_dir / "qrels.tsv"
    corpus_path = data_split_dir / "corpus.tsv"
    if not qrels_path.exists() or not corpus_path.exists():
        print(f"[SKIP] missing corpus/qrels in {data_split_dir}")
        return

    qid2docids = build_qid_to_docids_map(qrels_path)
    doc2detext = build_docid_to_detext_map(corpus_path)

    aligned: List[Tuple[str, str]] = []
    stats = {"total_rows": 0, "total_pairs": 0, "missing_docids": 0, "missing_detext": 0}

    for d in ("nds","bar","gsw"):
        fpath = step3_split_dir / f"{d}_dict_trans_queries.tsv"
        if not fpath.exists():
            print(f"[{data_split_dir.name}] warn: missing {fpath.name}, skip")
            continue
        rows = read_tsv_pairs(fpath)  # (qid, text)
        stats["total_rows"] += len(rows)
        for qid, text in rows:
            base_q = strip_variant(qid)
            docids = qid2docids.get(base_q, [])
            if not docids:
                stats["missing_docids"] += 1
                continue
            for docid in docids:
                detext = doc2detext.get(docid)
                if detext is None:
                    stats["missing_detext"] += 1
                    continue
                aligned.append((text, detext))
                stats["total_pairs"] += 1

    queries_out: List[Tuple[str, str]] = []
    corpus_out:  List[Tuple[str, str]] = []
    qrels_out_4col: List[Tuple[str, str, str, str]] = []

    for idx, (text, detext) in enumerate(aligned, start=1):
        sec_qid = f"{qid_prefix}{idx:0{pad}d}"
        sec_doc = f"{doc_prefix}{idx:0{pad}d}"
        queries_out.append((sec_qid, text))
        corpus_out.append((sec_doc, detext))
        qrels_out_4col.append((sec_qid, sec_doc))

    out_split_dir.mkdir(parents=True, exist_ok=True)
    write_tsv_pairs(out_split_dir / "de2dia_dict_queries.tsv", queries_out)
    write_tsv_pairs(out_split_dir / "de2dia_dict_corpus.tsv",  corpus_out)
    with (out_split_dir / "de2dia_dict_qrels.tsv").open("w", encoding="utf-8") as f:
        for sec_qid, sec_doc in qrels_out_4col:
            f.write(f"{sec_qid}\t{sec_doc}\t1\n")

    print(f"[{data_split_dir.name}] aligned={stats['total_pairs']} from rows={stats['total_rows']} "
          f"(missing_docids={stats['missing_docids']}, missing_detext={stats['missing_detext']}). "
          f"Wrote: {out_split_dir}/de2dia_dict_*.tsv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step3_root", required=True, help="Root of Step-2 outputs (contains <split>/<dial>_dict_trans_queries.tsv)")
    ap.add_argument("--data_root", required=True,  help="Root of original data (contains <split>/corpus.tsv and qrels.tsv)")
    ap.add_argument("--out_root",  required=True,  help="Output root for Step-3 files")
    ap.add_argument("--qid_prefix", default="QS", help="Prefix for newly assigned sec_qid")
    ap.add_argument("--doc_prefix", default="DS", help="Prefix for newly assigned sec_docid")
    ap.add_argument("--pad", type=int, default=7, help="Zero padding width for sec ids")
    args = ap.parse_args()

    step3_root = Path(args.step3_root)
    data_root  = Path(args.data_root)
    out_root   = Path(args.out_root)

    splits = []
    for name in ("train", "dev"):
        s3 = step3_root / name
        d0 = data_root  / name
        if s3.exists() and d0.exists():
            splits.append((s3, d0, out_root / name))
    if not splits:
        print(f"[ERR] No valid split pairs under {step3_root} and {data_root}")
        return

    for s3_dir, d0_dir, out_dir in splits:
        process_split(s3_dir, d0_dir, out_dir, args.qid_prefix, args.doc_prefix, args.pad)

    print(f"[OK] Step-5 finished. Outputs: {out_root}/<split>/de2dia_dict_*.tsv")

if __name__ == "__main__":
    main()
