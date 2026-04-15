DIALECT="${DIALECT:-gsw nds bar en}"        
K="${K:-10}"                                            
TAG="${TAG:-full.k100}"                                  
MODELS="${MODELS:-LaBSE Qwen3 BGE-M3 GTE-multilingual qwen3_ins ft_dict_LaBSE ft_dialemma_LaBSE ft_dialemma_qwen3 ft_gpt_LaBSE ft_gpt_qwen3 ft_ins_qwen3 ft_gpt_gte ft_dialemma_bge ft_gpt_bge BM25}" 
INTERSECT="${INTERSECT:-0}"                  


QR = "$BASE/data-1k100k-new/evaldatanew/${DIALECT}/qrels.tsv"
[[ -f "$QR" ]] || { echo "[ERR] missing qrels: $QR"; exit 2; }


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


CMD=( python "$BASE/retrieval/evaluate_baseline.py"
  --qrels "$QR"
  --runs "${RUNS[@]}"
  --k "$K"
  --out_csv "$OUT"
)


if [[ "$INTERSECT" == "1" ]]; then
  CMD+=( --intersect )
fi

printf "[CMD] %q " "${CMD[@]}"; echo
"${CMD[@]}"

echo "==== CSV HEAD (run,MRR@${K},Recall@${K},P@1) ===="
head -n 5 "$OUT" || true
echo "[OK] saved -> $OUT"
