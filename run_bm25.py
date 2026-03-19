"""
Lexical BM25 baseline

Example:
  BASE=$HOME/dialect-retrieval
  DIA=gsw
  TAG=full.k100

  python retrieval/run_bm25.py \
    --data_root "$BASE/data_eval_fixed99k/$DIA" \
    --topk 100 \
    --out "$BASE/runs/${DIA}.BM25.${TAG}.trec" \
    --run_name "BM25"
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


# read tsv files

def read_corpus_tsv(path: Path) -> Tuple[List[str], List[str]]:
    """
    corpus.tsv: doc_id \\t text
    """
    doc_ids, texts = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sp = line.split("\t", 1)
            if len(sp) != 2:
                continue
            did, txt = sp
            doc_ids.append(did)
            texts.append(txt)
    return doc_ids, texts


def read_queries_tsv(path: Path) -> Tuple[List[str], List[str]]:
    """
    queries.tsv: qid \\t text
    """
    qids, texts = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sp = line.split("\t", 1)
            if len(sp) != 2:
                continue
            qid, txt = sp
            qids.append(qid)
            texts.append(txt)
    return qids, texts


def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


# BM25 + TREC writer

def write_trec(
    out_path: Path,
    qids: List[str],
    doc_ids: List[str],
    top_idx: np.ndarray,
    top_scr: np.ndarray,
    run_name: str,
):

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for i, qid in enumerate(qids):
            for rank, (j, s) in enumerate(zip(top_idx[i], top_scr[i]), start=1):
                did = doc_ids[int(j)]
                w.write(f"{qid} Q0 {did} {rank} {float(s):.6f} {run_name}\n")


def main():
    ap = argparse.ArgumentParser(description="BM25 lexical baseline for dialect retrieval")
    # Data
    ap.add_argument("--data_root", default="", help="Dir with corpus.tsv & queries.tsv")
    ap.add_argument("--corpus", default="", help="Explicit path to corpus.tsv (doc_id \\t text)")
    ap.add_argument("--queries", default="", help="Explicit path to queries.tsv (qid \\t text)")
    # Retrieval config
    ap.add_argument("--topk", type=int, default=100, help="Top-k docs per query")
    # Output
    ap.add_argument("--out", required=True, help="Output TREC file")
    ap.add_argument("--run_name", default="BM25", help="Run tag in TREC file")
    args = ap.parse_args()

    if not args.corpus or not args.queries:
        if not args.data_root:
            raise SystemExit("Either --data_root or (--corpus and --queries) must be set.")
        root = Path(args.data_root)
        corpus_path = root / "corpus.tsv"
        queries_path = root / "queries.tsv"
    else:
        corpus_path = Path(args.corpus)
        queries_path = Path(args.queries)

    print(f"[LOAD] corpus: {corpus_path}")
    doc_ids, doc_texts = read_corpus_tsv(corpus_path)
    print(f"[LOAD] queries: {queries_path}")
    qids, q_texts = read_queries_tsv(queries_path)

    print(f"[STATS] |D|={len(doc_ids)}, |Q|={len(qids)}")

    # Build BM25 index
    print("[BM25] tokenizing corpus ...")
    tokenized_docs = [simple_tokenize(t) for t in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)
    print("[BM25] index ready.")

    all_top_idx = []
    all_top_scr = []
    K = min(args.topk, len(doc_ids))

    for qi, qtext in enumerate(q_texts):
        if (qi + 1) % 100 == 0:
            print(f"[BM25] processed {qi+1}/{len(q_texts)} queries ...", end="\r")
        q_tokens = simple_tokenize(qtext)
        scores = np.array(bm25.get_scores(q_tokens), dtype=np.float32)

        # partial top-k
        if K < len(scores):
            part = np.argpartition(-scores, K - 1)[:K]
            part = part[np.argsort(-scores[part])]
        else:
            part = np.argsort(-scores)
        all_top_idx.append(part)
        all_top_scr.append(scores[part])

    print("\n[BM25] search done for all queries.")

    top_idx = np.stack(all_top_idx, axis=0)
    top_scr = np.stack(all_top_scr, axis=0)

    out_path = Path(args.out)
    write_trec(out_path, qids, doc_ids, top_idx, top_scr, args.run_name)
    print(f"[OK] Wrote BM25 TREC run -> {out_path}")


if __name__ == "__main__":
    main()
