#!/usr/bin/env bash
#SBATCH -J barlen
#SBATCH --partition=kisski
#SBATCH -G A100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -C inet

set -euo pipefail
set -x

# 目录与日志
BASE="${PROJECT_DIR:-$HOME}/dialect-retrieval"
OUT="$BASE/runs_bar"
mkdir -p "$BASE/logs" "$BASE/runs_bar"
cd "$BASE"

# 激活环境
module purge && module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /user/jing.wang10/u21437/.conda/envs/llmdir-gpu || \
conda activate /user/jing.wang10/u21437/.conda/envs/llmdir


# 组装运行命令
python "$BASE/retrieval/ft_dense.py" \
  --model "$BASE/models_ft_dialemma_labse/labse.mixed" \
  --data_root "$BASE/data-1k100k-new/evaldatanew/bar" \
  --topk 100 \
  --batch 128 \
  --q_batch 128 \
  --use_faiss 0 \
  --out "$OUT/bar.ft_dialemma_labse.full.k100.trec" \
  --run_name "abzs"
  #--model "$BASE/models_ft_gpt_labse/labse.mixed" \
  #--data_root "$BASE/data-1k100k-new/evaldatanew/en" \

echo "[OK] run -> $OUT"
