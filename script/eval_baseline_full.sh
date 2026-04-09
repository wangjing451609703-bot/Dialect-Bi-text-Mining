#!/usr/bin/env bash
#SBATCH -J barleneval
#SBATCH --partition=kisski
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
set -x

# ==== 路径与环境 ====
BASE="${PROJECT_DIR:-$HOME}/dialect-retrieval"
mkdir -p "$BASE/logs" "$BASE/outputs_bar"
cd "$BASE"

module purge && module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
# 优先 GPU 环境；若没有则回退 llmdir
conda activate /user/jing.wang10/u21437/.conda/envs/llmdir-gpu || \
conda activate /user/jing.wang10/u21437/.conda/envs/llmdir

# ==== 参数（可用 sbatch --export 覆盖）====
DIALECT="${DIALECT:-gsw nds bar en 20 30 40 60 80 100}"                                # gsw / nds / bar
K="${K:-10}"                                             # 用于 MRR@K / Recall@K
TAG="${TAG:-full.k100}"                                  # 结果文件后缀，如 full.k100 / q100.k50.flat / ...
MODELS="${MODELS:-LaBSE Qwen3 BGE-M3 GTE-multilingual qwen3_ins ft_LaBSE ft_dict_LaBSE ft_dialemma_LaBSE ft_dialemma_qwen3  ft_gpt_LaBSE ft_gpt_qwen3 ft_ins_qwen3 ft_gpt_gte ft_dialemma_bge ft_gpt_bge ft_32_labse ft_16_labse ft_8_labse ft_32dia_labse BM25}"        # 评测哪些模型
INTERSECT="${INTERSECT:-0}"                              # 1=仅评 run∩qrels 的 qid（子集评测时设为1）

QR="$BASE/bar-wiki-len/${DIALECT}/qrels.tsv"
#QR="$BASE/data-1k100k-new/evaldatanew/${DIALECT}/qrels.tsv"
[[ -f "$QR" ]] || { echo "[ERR] missing qrels: $QR"; exit 2; }

# ==== 收集存在的 run 文件 ====
RUNS=()
for M in $MODELS; do
  R="$BASE/runs_bar/${DIALECT}.${M}.${TAG}.trec"
  if [[ -f "$R" ]]; then
    RUNS+=("$R")
  else
    echo "[WARN] skip missing run: $R"
  fi
done
(( ${#RUNS[@]} > 0 )) || { echo "[ERR] no runs found for dialect=$DIALECT tag=$TAG"; exit 3; }

OUT="$BASE/outputs_bar/${DIALECT}.${MODELS}.${TAG}.k${K}.mrr_rec_p1.csv"

# ==== 执行评测 ====
CMD=( python "$BASE/retrieval/evaluate_baseline.py"
  --qrels "$QR"
  --runs "${RUNS[@]}"
  --k "$K"
  --out_csv "$OUT"
)

# 只在子集评测时开启
if [[ "$INTERSECT" == "1" ]]; then
  CMD+=( --intersect )
fi

printf "[CMD] %q " "${CMD[@]}"; echo
"${CMD[@]}"

echo "==== CSV HEAD (run,MRR@${K},Recall@${K},P@1) ===="
head -n 5 "$OUT" || true
echo "[OK] saved -> $OUT"
