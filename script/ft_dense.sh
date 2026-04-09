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
