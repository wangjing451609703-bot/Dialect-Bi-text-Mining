"""
Zero-shot dense retrieval (supports finetuned local models).

Usage examples:
python retrieval/zero_shot_dense_fast.py \
  --model /path/to/your_finetuned_model_dir \
  --data_root $BASE/data_eval_fixed99k/gsw \
  --topk 100 --batch 256 --q_batch 128 --fp16 1 --use_faiss 0 \
  --out $BASE/runs/gsw.labse_ft.k100.trec

If you prefer explicit files:
  --corpus path/to/corpus.tsv --queries path/to/queries.tsv
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from sentence_transformers import SentenceTransformer

# Model map
MODEL_MAP = {
    "LaBSE": "sentence-transformers/LaBSE",
    "BGE-M3": "BAAI/bge-m3",
    "GTE-multilingual": "Alibaba-NLP/gte-multilingual-base",
    "Qwen3-Embedding": "Qwen/Qwen3-Embedding-0.6B"
}

# Local finetuned model directory, return a local dir if exists, else load from hf
def resolve_model(name_or_path: str) -> str:
    p = Path(name_or_path)
    if p.exists():  
        return str(p)
    if name_or_path in MODEL_MAP:
        return MODEL_MAP[name_or_path]
    return name_or_path

# read tsv
def read_corpus_tsv(path: Path) -> Tuple[List[str], List[str]]:
    """
    corpus.tsv: doc_id \t text
    """
    doc_ids, texts = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            sp = line.split("\t", 1)
            if len(sp) != 2: continue
            did, txt = sp[0], sp[1]
            doc_ids.append(did)
            texts.append(txt)
    return doc_ids, texts

def read_queries_tsv(path: Path) -> Tuple[List[str], List[str]]:
    """
    queries.tsv: qid \t text
    """
    qids, texts = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            sp = line.split("\t", 1)
            if len(sp) != 2: continue
            qid, txt = sp[0], sp[1]
            qids.append(qid)
            texts.append(txt)
    return qids, texts

# Embedding
def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def load_st_model(arg_model: str, fp16: bool) -> SentenceTransformer:
    resolved = resolve_model(arg_model)
    model = SentenceTransformer(resolved, trust_remote_code=True)
    if fp16 and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    return model

@torch.inference_mode()
# Encode to np float32.
def encode_texts(model: SentenceTransformer, texts: List[str], batch: int, device: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    emb = model.encode(
        texts,
        batch_size=batch,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
        device=device
    )
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32, copy=False)
    return emb

# Search backends
def search_numpy(
    q_emb: np.ndarray, d_emb: np.ndarray, topk: int, q_batch: int
) -> Tuple[np.ndarray, np.ndarray]:
    N = d_emb.shape[0]
    tk = min(topk, N)
    all_idx = []
    all_scr = []
    for i in range(0, q_emb.shape[0], q_batch):
        Q = q_emb[i:i+q_batch]  
        sims = Q @ d_emb.T      
        # partial top-k
        part = np.argpartition(-sims, kth=tk-1, axis=1)[:, :tk]
        part_scores = np.take_along_axis(sims, part, axis=1)
        # sort top-k
        order = np.argsort(-part_scores, axis=1)
        top_idx = np.take_along_axis(part, order, axis=1)
        top_scr = np.take_along_axis(part_scores, order, axis=1)
        all_idx.append(top_idx)
        all_scr.append(top_scr)
    return np.vstack(all_idx), np.vstack(all_scr)

def build_faiss_index(d_emb: np.ndarray, use_ivf: int = 0, nlist: int = 4096) -> "faiss.Index":
    dim = d_emb.shape[1]
    if use_ivf:
        # inner product
        quantizer = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(d_emb)
        idx.add(d_emb)
    else:
        idx = faiss.IndexFlatIP(dim)
        idx.add(d_emb)
    return idx

def search_faiss(
    q_emb: np.ndarray,
    d_emb: np.ndarray,
    topk: int,
    ivf_nlist: int = 4096,
    ivf_nprobe: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_FAISS:
        raise RuntimeError("faiss not available. Install faiss-cpu or set --use_faiss 0.")
    N = d_emb.shape[0]
    tk = min(topk, N)
    use_ivf = int(ivf_nlist > 0 and N >= ivf_nlist)
    idx = build_faiss_index(d_emb, use_ivf=use_ivf, nlist=ivf_nlist)
    if use_ivf:
        idx.nprobe = max(1, ivf_nprobe)
    scores, indices = idx.search(q_emb, tk)
    return indices, scores

# write trec
def write_trec(
    out_path: Path,
    qids: List[str],
    doc_ids: List[str],
    top_idx: np.ndarray,
    top_scr: np.ndarray,
    run_name: str
):

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for i, qid in enumerate(qids):
            for r, (j, s) in enumerate(zip(top_idx[i], top_scr[i]), start=1):
                did = doc_ids[int(j)]
                w.write(f"{qid} Q0 {did} {r} {float(s):.6f} {run_name}\n")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Zero-shot dense retrieval (local finetuned model supported)")
    # Data
    ap.add_argument("--data_root", default="", help="Directory containing corpus.tsv & queries.tsv")
    ap.add_argument("--corpus", default="", help="Path to corpus.tsv (doc_id \\t text)")
    ap.add_argument("--queries", default="", help="Path to queries.tsv (qid \\t text)")
    # Model
    ap.add_argument("--model", required=True, help="Model alias/HF name or local directory of finetuned model")
    ap.add_argument("--fp16", type=int, default=1, help="1=encode with AMP on GPU if available")
    # Retrieval config
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256, help="Doc encoding batch size")
    ap.add_argument("--q_batch", type=int, default=128, help="Query encoding batch size")
    ap.add_argument("--use_faiss", type=int, default=0, help="1=FAISS CPU, 0=NumPy")
    ap.add_argument("--faiss_ivf", type=int, default=4096, help="IVF nlist (0=disable IVF)")
    ap.add_argument("--faiss_nprobe", type=int, default=32)
    # Output
    ap.add_argument("--out", required=True, help="Output TREC file")
    ap.add_argument("--run_name", default="zs_dense", help="Run tag in trec file")
    args = ap.parse_args()

    # Resolve data paths
    if args.data_root:
        corpus_path = Path(args.data_root) / "corpus.tsv"
        queries_path = Path(args.data_root) / "queries.tsv"
    else:
        corpus_path = Path(args.corpus)
        queries_path = Path(args.queries)


    # Load data
    doc_ids, doc_texts = read_corpus_tsv(corpus_path)
    qids, q_texts = read_queries_tsv(queries_path)
    print(f"[LOAD] docs={len(doc_ids)} queries={len(qids)}")

    # Load model
    model = load_st_model(args.model, bool(args.fp16))
    device = "cuda" if (torch.cuda.is_available()) else "cpu"
    print(f"[MODEL] {resolve_model(args.model)} on {device}  (fp16={bool(args.fp16)})")

    # Encode
    d_emb = encode_texts(model, doc_texts, batch=args.batch, device=device)
    q_emb = encode_texts(model, q_texts, batch=args.q_batch, device=device)
    print(f"[EMB] d_emb={d_emb.shape} q_emb={q_emb.shape} (L2-normalized)")

    # Search
    if args.use_faiss:
        if not HAS_FAISS:
            raise RuntimeError("faiss not installed. Use --use_faiss 0 or install faiss-cpu.")
        top_idx, top_scr = search_faiss(
            q_emb, d_emb, args.topk, ivf_nlist=args.faiss_ivf, ivf_nprobe=args.faiss_nprobe
        )
        print(f"[SEARCH] FAISS (IVF nlist={args.faiss_ivf}, nprobe={args.faiss_nprobe}) done.")
    else:
        top_idx, top_scr = search_numpy(q_emb, d_emb, args.topk, q_batch=max(1, args.q_batch))
        print("[SEARCH] NumPy matmul done.")

    # Write trec
    out_path = Path(args.out)
    write_trec(out_path, qids, doc_ids, top_idx, top_scr, args.run_name)
    print(f"[OK] Wrote TREC to: {out_path}")

if __name__ == "__main__":
    main()
