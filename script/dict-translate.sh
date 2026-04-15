BASE="${PROJECT_DIR:-$HOME}/dialect-retrieval"   # change to your base path

STEP1="$BASE/data-dict-split"          
DATA="$BASE/data_dict_mixed"           
OUT1="$BASE/data_dict/bar2de"  # change to your out path for bar-de translated sentences
OUT2="$BASE/data_dict/de2bar"  # change to your out path for de-bar translated sentences
OUT3="$BASE/data_dict/align/de-bar"  # change to your out path for aligned de-bar pairs (as query side)
OUT4="$BASE/data_dict/align/bar-de"  # change to your out path for aligned bar-de pairs (as document side)
OUT5="$BASE/data_dict/all_dialemma_translated" # change to your out path for merged bar-de & de-bar pairs (as full query-document pool)
OUT_Final="$BASE/data_ft_dialemma"  # change to your out path for final train/dev set
DIALEMMA="$BASE/dicts"  # change to your dialemma dictionary path                



python "$BASE/dict_base/bar2de.py" \
  --step1_root "$STEP1" \
  --data_root  "$DATA" \
  --out_root   "$OUT1" \
  --dialemma_path  "$DIALEMMA/dialemma.jsonl" \
  --lowercase_match \
  --keep_if_replaced_only
echo "[DONE] bar→de dictionary translation finished -> $OUT1."


python "$BASE/dict_base/de2bar.py" \
  --data_root  "$DATA" \
  --out_root   "$OUT2" \
  --dialemma_path  "$DIALEMMA/dialemma.jsonl" \
  --lowercase_match \
  --max_expansions 50
echo "[DONE] de→bar dictionary translation finished -> $OUT2."


python "$BASE/dict_base/de2dia_align.py" \
  --step3_root "$OUT2" \
  --data_root  "$DATA" \
  --out_root   "$OUT3" \
  --qid_prefix QS \
  --doc_prefix DS \
  --pad 7
echo "[DONE] de→dia alignment and re-indexing finished -> $OUT3."


python "$BASE/dict_base/dia2de_align.py" \
  --step2_root "$OUT1" \
  --data_root  "$DATA" \
  --out_root   "$OUT4" \
  --qid_prefix SQ \
  --doc_prefix SD \
  --pad 7
echo "[DONE] dia→de alignment and re-indexing finished -> $OUT4."


python "$BASE/dict_base/unify.py" \
  --step4_root "$OUT4" \
  --step5_root  "$OUT3" \
  --out_root   "$OUT5" 
echo "[DONE] Step 6 merged & identicals removed successfully."


python "$BASE/dict_base/select.py" \
  --in_root "$OUT5/train" \
  --out_root   "$OUT_Final" \
  --n_train  32458 \
  --n_dev    3605 \
  --seed     2025
echo "[DONE] Step 7 succeed."
