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
