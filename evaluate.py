import argparse, csv
from pathlib import Path
from collections import defaultdict
import numpy as np

# Read trec
def load_qrels(path: str):
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4: 
                continue
            qid, _, docid, rel = row[0], row[1], row[2], int(row[3])
            qrels[qid][docid] = rel
    return qrels

def load_run_trec(path: str):
    run = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6: 
                continue
            qid, docid, score = p[0], p[2], float(p[4])
            run[qid][docid] = score
    return run

# MRR
def mrr_at_k(qrels, run, k=10):
    vals=[]
    for qid, rels in qrels.items():
        items = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
        rr=0.0
        for rank,(doc,_) in enumerate(items[:k], start=1):
            if rels.get(doc,0)>0:
                rr = 1.0/rank
                break
        vals.append(rr)
    return float(np.mean(vals) if vals else 0.0)

# Recall
def recall_at_k(qrels, run, k=10):
    hits=0; total=len(qrels)
    for qid, rels in qrels.items():
        items = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])[:k]
        hit = any(rels.get(doc,0)>0 for doc,_ in items)
        hits += int(hit)
    return float(hits/total) if total else 0.0

# P1
def precision_at_1(qrels, run):
    vals=[]
    for qid, rels in qrels.items():
        items = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
        if items:
            top1_doc = items[0][0]
            vals.append(1.0 if rels.get(top1_doc,0)>0 else 0.0)
        else:
            vals.append(0.0)
    return float(np.mean(vals) if vals else 0.0)

def main():
    ap = argparse.ArgumentParser(description="Evaluate MRR@K / Recall@K / P@1 over runs")
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=10, help="K used for MRR@K & Recall@K")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--intersect", action="store_true",
                    help="Only evaluate on intersection of run qids and qrels qids (for subset runs).")
    args = ap.parse_args()

    
    base_qrels = load_qrels(args.qrels)

    rows=[]  # (run_name, MRR@K, Recall@K, P@1)

    for run_path in args.runs:
        p = Path(run_path)
        run = load_run_trec(run_path)

        if args.intersect:
            qids = set(run.keys()) & set(base_qrels.keys())
            qrels = {qid: base_qrels[qid] for qid in qids}
        else:
            qrels = base_qrels

        n_qrels = len(qrels)
        n_runq  = len(run.keys() & qrels.keys())
        print(f"[INFO] evaluating {p.stem}: qrels_q={n_qrels}, run_q_in_qrels={n_runq}, intersect={args.intersect}")

        mrr = mrr_at_k(qrels, run, k=args.k)
        rec = recall_at_k(qrels, run, k=args.k)
        p1  = precision_at_1(qrels, run)

        rows.append((p.stem, float(mrr), float(rec), float(p1)))

    if not rows:
        raise SystemExit("[ERR] no valid runs to evaluate.")

    # sort by MRR@K desc
    rows.sort(key=lambda r: r[1], reverse=True)
    header = ["run", f"MRR@{args.k}", f"Recall@{args.k}", "P@1"]
    colw = [max(len(header[i]), max(len(str(rows[j][i])) for j in range(len(rows)))) for i in range(4)]
    fmt = "  ".join(["{:<" + str(colw[0]) + "}", "{:>" + str(colw[1]) + "}", "{:>" + str(colw[2]) + "}", "{:>" + str(colw[3]) + "}"])
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}"))

    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])
        print(f"[OK] saved -> {outp}")

if __name__ == "__main__":
    main()
