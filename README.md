# Dialect-Bi-text-Mining

This repository contains the code used in the thesis project on **Bi-Text Mining Across German Dialects: A New Benchmark for Dialect–Standard German Translation Retrieval**, including:

- **Benchmark construction** with 1k-100k evaluation set.
- **Zero-shot baselines**: dense bi-encoders + BM25.
- **Fine-tuning** bi-encoders with **MultipleNegativesRankingLoss (MNRL)**.
- **Evaluation** using MRR@10, Recall@10 and Precision@1 on first-stage dense retrieval.
- **Synthetic data generation (dictionary-based)** for Bavarian (bar) using the Dialemma lexicon.
- **Synthetic data generation (LLM-based)** for Low German(nds), Alemannic(gsw)and Bavarian (bar) using GPT-4o.

The project targets dialects such as **Alemannic (als)**, **Low German (nds)**, and **Bavarian (bar)**, consistent with the thesis scope.


---
## Data Loading

Download the raw data from **`Tatoeba/`,`WikiMatrix/`,`wikimedia/`** at:

- bar-de:https://opus.nlpl.eu/corpora-search/bar&de
- nds-de:https://opus.nlpl.eu/corpora-search/nds&de
- gsw-de:https://opus.nlpl.eu/corpora-search/gsw&de
- swg-de:https://opus.nlpl.eu/corpora-search/swg&de
- en-de:https://opus.nlpl.eu/corpora-search/en&de

Or use the `data/` folder for preprocessed data:

- `data-1k-100k-new/`: evaluation set
- `data_ft_dialemma/`: dictionary-based synthetic data set
- `data-ft_gpt/`: LLM-generated synthetic data set

Download the dialect dictionaries used in this project from:
- `Dialemma`: https://github.com/mainlp/dialemma?tab=readme-ov-file


---
### Baseline Models

Use `SentenceTransformer` to load baseline bi-encoder models.

- LaBSE: sentence-transformers/LaBSE
- BGE-M3: BAAI/bge-m3
- GTE-multilingual: Alibaba-NLP/gte-multilingual-base
- Qwen3-Embedding: Qwen/Qwen3-Embedding-0.6

Use `rank-bm25` to load BM-25

---
## Requirements

Python 3.10

Pytorch >= 12.1

pip install sentence-transformers faiss-cpu numpy pytrec_eval rank-bm25


---
## Preprocessed Dataset Construction

**1.Eval Dataset Construction**

Use `code/1k-100k eval data/1k100k.ipynb` and `code/1k-100k eval data/evalset.ipynb` to create 1k-100k evaluation set.

**2.Dictionary-based Synthetic Dataset Construction**

Use `code/dict-base/` folder to create this fine-tuning dataset step by step:

- `00_load_dataset.py`
- `01_split_by_dialect.py`
- `02_de2bar.py`
- `02_bar2de.py`
- `03_dia2de_align.py`
- `03_de2dia_align.py`
- `04_unify.py`
- `05_select.py`

**3.LLM-generated Synthetic Dataset Construction**

Use `de-dialect-gpt.ipynb` to translate de sentences to dialects. `OpenAi_API_Key` is required.


---
## Running Experiment

**1.Zero-shot Baseline**

Run `dense_retrieval.py` for zero-shot baseline. Save the output `*.trec` file name in the format of: `{*DIALECT}.{*MODELS}.-full.k100.mrr_rec_p1.trec`.

For BM-25 baseline, run `run_bm25.py`

**2.Fine-tuning**

Use `train.py` to fine-tune models on synthetic dataset.

Example usage:

```bash
BASE=$PWD
TRAIN_DIR="$BASE/data_train_synth/bar/train"
DEV_DIR="$BASE/data_train_synth/bar/dev"
OUT_DIR="$BASE/model/labse_ft_bar_de"

python retrieval/train_opus.py \
  --model LaBSE \
  --train_dir "$TRAIN_DIR" \
  --dev_dir "$DEV_DIR" \
  --output_dir "$OUT_DIR" \
  --epochs 1 \
  --batch_size 16 \
  --lr 2e-5 \
  --fp16 1 \
  --eval_steps 1000
```

**3.Evaluation**

Use `evaluate.py` to get MRR@10, Recall@10 and Precision@1 scores.

Example usage:
```bash
BASE=$PWD

python retrieval/evaluate_baseline.py \
  --qrels "$BASE/data_eval_fixed99k/gsw/qrels.tsv" \
  --runs \
    "$BASE/runs/gsw.LaBSE.-full.k100.mrr_rec_p1.trec" \
    "$BASE/runs/gsw.BM25.-full.k100.mrr_rec_p1.trec" \
  --k 10 \
  --out_csv "$BASE/outputs/gsw.k10.csv"
```


---
## Acknowledgements

Code in `dense_retrieval.py` and `run_bm25.py` are partly adapted from:

- **Evaluating Large Language Models for Cross-Lingual Information Retrieval** — https://github.com/mainlp/llm-clir

Use of AI:

- AI tools were used during development to assist with debugging, refactoring and drafting small code components.
