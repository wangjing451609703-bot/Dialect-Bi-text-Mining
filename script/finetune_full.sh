#!/usr/bin/env bash
#SBATCH -J ft_dia_qwen3
#SBATCH --partition=kisski
#SBATCH -G A100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -C inet

set -euo pipefail
set -x

BASE="${PROJECT_DIR:-$HOME}/dialect-retrieval"
DATA="$BASE/data_ft_dialemma"  #data_ft_dialemma data-ft-gpt
OUT="$BASE/models_ft_dialemma_qwen3/qwen3.mixed"
mkdir -p "$BASE/logs" "$OUT"

# 激活环境
module purge && module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /user/jing.wang10/u21437/.conda/envs/llmdir-gpu || \
conda activate /user/jing.wang10/u21437/.conda/envs/llmdir


python "$BASE/finetune/train_opus.py" \
  --train_dir "$DATA/train" \
  --dev_dir   "$DATA/dev" \
  --model     Qwen3-Embedding \
  --out_dir   "$OUT" \
  --epochs    3 \
  --batch_size 64 \
  --lr        2e-5 \
  --eval_steps 1000 \
  --fp16 1 \
  --safe_serialization 1

echo "[OK] LaBSE finetune completed -> $OUT"
