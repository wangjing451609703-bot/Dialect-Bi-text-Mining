#!/usr/bin/env bash
#SBATCH -J run_bm25
#SBATCH --partition=kisski
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
set -x

BASE="${PROJECT_DIR:-$HOME}/dialect-retrieval"
mkdir -p "$BASE/logs" "$BASE/runs_bar"
cd "$BASE"

module purge && module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /user/jing.wang10/u21437/.conda/envs/llmdir-gpu || \
conda activate /user/jing.wang10/u21437/.conda/envs/llmdir



DATA_ROOT="$BASE/bar-wiki"
RUNS_DIR="$BASE/runs_bar"
TAG="full.k100"   


python retrieval/run_bm25.py \
    --data_root "$DATA_ROOT" \
    --topk 100 \
    --out "$RUNS_DIR/bar.BM25.${TAG}.trec" \
    --run_name "BM25"

echo "[DONE] BM25 finished for bar"