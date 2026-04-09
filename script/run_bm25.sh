DATA_ROOT="$BASE/1k-100k eval data/evaldatanew/bar"
RUNS_DIR="$BASE/runs_bar"
TAG="full.k100"   


python retrieval/run_bm25.py \
    --data_root "$DATA_ROOT" \
    --topk 100 \
    --out "$RUNS_DIR/bar.BM25.${TAG}.trec" \
    --run_name "BM25"

echo "[DONE] BM25 finished for bar"
