"""
Finetune dialect→German retrieval with Sentence-Transformers:
- MultipleNegativesRankingLoss
- Per-epoch dev evaluation (MRR@10 / Recall@10 / Precision@1)
- Also prints Train Eval Loss & Dev Eval Loss (MNRL, no-grad)
- Dev evaluation uses merged corpus: train_corpus ∪ dev_corpus
- --save_best_only: save model only when dev MRR@10 improves
- Auto-clean ckpts/ unless --keep_ckpts
"""

import argparse
import os
import shutil
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    losses,
    InputExample,
    LoggingHandler,
)


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Model map 
MODEL_MAP = {
    "LaBSE": "sentence-transformers/LaBSE",
    "BGE-M3": "BAAI/bge-m3",
    "GTE-multilingual": "Alibaba-NLP/gte-multilingual-base",
    "Qwen3-Embedding": "Qwen/Qwen3-Embedding-0.6B",
}


# Load data
def hid(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def read_tsv_pairs(corpus_tsv: Path, queries_tsv: Path, qrels_tsv: Path):

    corpus, queries, rel = {}, {}, {}

    with open(corpus_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            did, txt = line.split("\t", 1)
            corpus[did] = txt

    with open(queries_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            qid, txt = line.split("\t", 1)
            queries[qid] = txt

    with open(qrels_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                qid, did, _ = parts
            elif len(parts) == 4:
                qid, _, did, _ = parts
            else:
                # ignore error lines
                continue
            rel.setdefault(qid, {})[did] = 1

    return corpus, queries, rel


def build_pair_dataset(dir_path: Path) -> List[InputExample]:
    corpus, queries, rel = read_tsv_pairs(
        dir_path / "corpus.tsv",
        dir_path / "queries.tsv",
        dir_path / "qrels.tsv",
    )
    examples = []
    for qid, pos in rel.items():
        if qid not in queries:
            continue
        qtxt = queries[qid]
        did = next(iter(pos.keys()))
        dtxt = corpus.get(did)
        if dtxt is None:
            continue
        examples.append(InputExample(texts=[qtxt, dtxt]))
    return examples


def build_train_dataset(train_dir: Path):
    return build_pair_dataset(train_dir)


def load_model(name_or_path: str, fp16: bool):
    model_name = MODEL_MAP.get(name_or_path, name_or_path)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    # avoid OOM
    model.max_seq_length = 128
    try:
        tr = model[0].auto_model
        tr.config.use_cache = False
        tr.gradient_checkpointing_enable()
    except Exception as e:
        print("[WARN] cannot enable grad checkpointing:", e)
    if fp16 and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    return model



# IR evaluation
def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _encode_texts(
    model,
    texts: List[str],
    batch_size: int = 256,
) -> np.ndarray:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            v = model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,  
                device=device,
            )
        v = v.astype(np.float32, copy=False)
        embs.append(v)
    if not embs:
        return np.zeros((0, 768), dtype=np.float32)
    X = np.vstack(embs)
    X = _l2_normalize(X)
    return X


def eval_ir_metrics(
    model,
    corpus: Dict[str, str],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
    batch_size: int = 64,
) -> Dict[str, float]:

    doc_ids = list(corpus.keys())
    q_ids = list(queries.keys())

    doc_texts = [corpus[d] for d in doc_ids]
    qry_texts = [queries[q] for q in q_ids]

    # encode
    D = _encode_texts(model, doc_texts, batch_size=batch_size)
    Q = _encode_texts(model, qry_texts, batch_size=batch_size)

    if D.shape[0] == 0 or Q.shape[0] == 0:
        return {"MRR@10": 0.0, "Recall@10": 0.0, "Precision@1": 0.0}

    topk = min(k, D.shape[0])
    mrr, recall, p1 = 0.0, 0.0, 0.0

    # compute query by query
    for qi, qvec in enumerate(Q):
        sims = D @ qvec  # dot == cosine
        # top-k
        top_idx = np.argpartition(-sims, topk - 1)[:topk]
        top_sorted = top_idx[np.argsort(-sims[top_idx])]
        ranked_docids = [doc_ids[i] for i in top_sorted]

        pos_set = set(qrels.get(q_ids[qi], {}).keys())

        # Precision@1
        p1 += 1.0 if ranked_docids and ranked_docids[0] in pos_set else 0.0
        # Recall@10
        hit = any(d in pos_set for d in ranked_docids)
        recall += 1.0 if hit else 0.0
        # MRR@10
        rr = 0.0
        for r, d in enumerate(ranked_docids, start=1):
            if d in pos_set:
                rr = 1.0 / r
                break
        mrr += rr

    n = max(1, len(Q))
    return {
        "MRR@10": mrr / n,
        "Recall@10": recall / n,
        "Precision@1": p1 / n,
    }


# Save best model by mrr
def pick_dev_score(results: Dict[str, float]) -> float:

    return float(results.get("MRR@10", 0.0))


def print_eval_table(results: Dict[str, float]):

    print("----- Dev Evaluation Results -----")
    print(f"MRR@10: {results.get('MRR@10', 0.0):.4f}")
    print(f"Recall@10: {results.get('Recall@10', 0.0):.4f}")
    print(f"Precision@1: {results.get('Precision@1', 0.0):.4f}")
    print("----------------------------------")


def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# Eval loss (MNRL, no-grad, min_bs=2) 
@torch.no_grad()
def eval_pair_loss(model: SentenceTransformer, examples: List[InputExample], batch_size: int = 64) -> float:

    if not examples:
        return 0.0

    bs = max(2, min(batch_size, len(examples)))
    # smart batching，InputExample -> (features, labels)
    loader = DataLoader(
        examples,
        batch_size=bs,
        shuffle=False,
        drop_last=True, 
        collate_fn=model.smart_batching_collate
    )

    loss_fn = losses.MultipleNegativesRankingLoss(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total, cnt = 0.0, 0
    for features, labels in loader:
        features = [{k: v.to(device) for k, v in f.items()} for f in features]
        if labels is not None and hasattr(labels, "to"):
            labels = labels.to(device)

        loss_val = loss_fn(features, labels)
        total += float(loss_val.item())
        cnt += 1

    if cnt == 0:
        return 0.0
    return total / cnt



# Training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--dev_dir", required=True)
    ap.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()) + ["custom"])
    ap.add_argument("--model_path", default="")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64) # 32,16,8 for ablation
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--fp16", type=int, default=1)

    ap.add_argument("--keep_ckpts", action="store_true", help="save ckpts；delete after training")
    ap.add_argument("--safe_serialization", type=int, default=1, help="1= save as safetensors")

    args = ap.parse_args()

    model_id = args.model_path if args.model == "custom" else args.model
    model = load_model(model_id, fp16=bool(args.fp16))

    train_dir = Path(args.train_dir)
    dev_dir = Path(args.dev_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    train_samples = build_pair_dataset(train_dir)
    dev_samples   = build_pair_dataset(dev_dir)

    train_loader = DataLoader(
        train_samples, shuffle=True, batch_size=args.batch_size, drop_last=True
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Build merged dev corpus
    dev_corpus, dev_queries, dev_rel = read_tsv_pairs(
        dev_dir / "corpus.tsv", dev_dir / "queries.tsv", dev_dir / "qrels.tsv"
    )
    train_corpus_only, _, _ = read_tsv_pairs(
        train_dir / "corpus.tsv", train_dir / "queries.tsv", train_dir / "qrels.tsv"
    )
    merged_corpus = dict(train_corpus_only)  # train docs as distractors
    merged_corpus.update(dev_corpus)         # ensure dev positives exist

    print(
        f"[INFO] TrainSamples={len(train_samples)} | "
        f"DevSamples={len(dev_samples)} | "
        f"DevQueries={len(dev_queries)} | "
        f"DevDocs(merged)={len(merged_corpus)}"
    )

    warmup_steps = int(len(train_loader) * args.epochs * args.warmup_ratio)
    ckpt_path = out_dir / "ckpts"


    # Training epochs
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== EPOCH {epoch}/{args.epochs} =====")
        fit_kwargs = dict(
            train_objectives=[(train_loader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps,
            use_amp=bool(args.fp16),
            optimizer_params={"lr": args.lr},
            checkpoint_path=str(ckpt_path),
            checkpoint_save_steps=args.eval_steps,
            show_progress_bar=True,
            output_path=str(out_dir)
        )

        model.fit(**fit_kwargs)

        # Dev evaluation
        print(f"[EVAL] Evaluating on dev (merged corpus) after epoch {epoch} ...")
        results = eval_ir_metrics(
            model=model,
            corpus=merged_corpus,
            queries=dev_queries,
            qrels=dev_rel,
            k=10,
            batch_size=64,
        )
        print_eval_table(results)
        score = pick_dev_score(results)

        # Eval losses
        train_eval_loss = eval_pair_loss(model, train_samples, batch_size=min(256, max(2, args.batch_size)))
        dev_eval_loss   = eval_pair_loss(model, dev_samples,   batch_size=min(256, max(2, args.batch_size)))
        print(f"[LOSS] Train Eval Loss={train_eval_loss:.6f} | Dev Eval Loss={dev_eval_loss:.6f}")



    # save last model
    model.save(str(out_dir), safe_serialization=bool(args.safe_serialization))
    print(f"[FINAL] Saved LAST model to {out_dir}")

    # delete ckpts (save memory space)
    if not args.keep_ckpts and ckpt_path.is_dir():
        shutil.rmtree(ckpt_path, ignore_errors=True)
        print(f"[CLEANUP] Removed checkpoint directory: {ckpt_path}")


if __name__ == "__main__":
    main()
