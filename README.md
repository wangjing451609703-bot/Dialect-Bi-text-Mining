# Dialect-Bi-text-Mining

This repository contains the code used in the thesis project on **Bi-Text Mining Across German Dialects: A New Benchmark for Dialect–Standard German Translation Retrieval**, including:

- **Benchmark construction** with 1k-100k evaluation set.
- **Zero-shot baselines**: dense bi-encoders + BM25.
- **Fine-tuning** bi-encoders with **MultipleNegativesRankingLoss (MNRL)**.
- **Evaluation** using MRR@10, Recall@10 and Precision@1 on first-stage dense retrieval.
- **Synthetic data generation (dictionary-based)** for Bavarian (bar) using the Dialemma lexicon.
- **Synthetic data generation (LLM-based)** for Low German(nds), Alemannic(gsw)and Bavarian (bar) using GPT-4o.

The project targets dialects such as **Alemannic (als/gsw+swg)**, **Low German (nds)**, and **Bavarian (bar)**, consistent with the thesis scope.


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
- `wiki-data.rar`: evaluation set extracted from **`WikiMatrix/`** and **`wikimedia/`** for data quality analysis

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
### Fine-tuned Models

You can download our fine-tuned models from: https://huggingface.co/JW451609703/models. Or you can also train the models locally use the scripts.

---
## Requirements

Install dependencies:

```
pip install -U sentence-transformers faiss-cpu openai tqdm pandas pytrec_eval rank-bm25 transformers
```

See `requirements.txt` for details. 

For `GTE-multilingual`, we recommand following versions:

```
pip install -q "sentence-transformers<3.0" "transformers>=4.36,<4.50" 
```


---
## Preprocessed Dataset Construction

**1.Eval Dataset Construction**

Use `code/1k-100k eval data/1k100k.py` and `code/1k-100k eval data/evalset.py` to create 1k-100k evaluation set. Or use `data/data-1k-100k-new/` for convenience.

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

Or use `data/data_ft_dialemma/` for convenience.

**3.LLM-generated Synthetic Dataset Construction**

Use `de-dialect-gpt.py` to translate de sentences to dialects. `OpenAi_API_Key` is required. Or use `data/data-ft_gpt/` for convenience.



---
## Running Experiment

### Getting results for zero-shot baselines, BM25 and fine-tuning (in Table 2, 3 and 4):

**1. data quality analysis (Table 2)**

Run `run_bm25.py` or `script/run_bm25.sh`. Save the output `*.trec` file name in the format of: `{*DIALECT}.BM25.-full.k100.mrr_rec_p1.trec`.

Example usage:

```
bash run_bm25.sh --data_root "<wiki-data root>"
```

Run `evaluate.py` or `script/eval.sh` to get MRR@10, Recall@10 and Precision@1 scores.

Example useage:

```
bash zs.sh --export DIALECT=bar, K=10, TAG=full.k100, MODEL=BM25, INTERSECT=0
```

Valid choices for "DIALECT" param are `bar, nds, gsw`.


**2.Zero-shot Baseline (Table 3 "Lexical Retrieval Baseline" and "Zero-Shot Evaluation", Table 4)**

Run `dense_retrieval.py` or `script/zs.sh` for zero-shot baseline. Save the output `*.trec` file name in the format of: `{*DIALECT}.{*MODELS}.-full.k100.mrr_rec_p1.trec`.

Example useage:

```
bash zs.sh
--model <model name> \
--data_root <data root path> \
--out <out path>
```

For BM-25 baseline, run `run_bm25.py` or `script/run_bm25.sh`. Change the `data_root` param to `data/data-1k100k-new`.

Run `evaluate.py` or `script/eval.sh` to get MRR@10, Recall@10 and Precision@1 scores.

Example useage:

```
bash zs.sh --export DIALECT=bar, K=10, TAG=full.k100, MODEL=LaBSE, INTERSECT=0
```

Valid choices for "DIALECT" param are `bar, nds, gsw, en`, for  "MODEL" are `LaBSE, GTE-multilingual, BGE-M3, Qwen3-Embedding, BM25`.

**3.Fine-tuning (Table 3 "Fine-tuning on dictionary-based/LLM-generated translations")**

Use `train.py` or `script/finetune_full.sh` to fine-tune models on synthetic dataset.

Example useage:

```
bash finetune_full.sh
  --train_dir <training data path> \
  --dev_dir   <dev data path> \
  --model     <model name> \
  --out_dir   <out path> \
```
Run `dense_retrieval.py` or `script/zs.sh` to get the `*.trec` file. For the `--model` param, example could be `$BASE/out path of finetune_full.sh/labse.mixed`.

Run `evaluate.py` or `script/eval.sh` to get MRR@10, Recall@10 and Precision@1 scores.

Example useage:

```
bash zs.sh --export DIALECT=bar, K=10, TAG=full.k100, MODEL=ft_gpt_LaBSE, INTERSECT=0
```

Valid choices for "MODEL" param are `ft_dict_LaBSE, ft_dialemma_LaBSE, ft_dialemma_qwen3, ft_gpt_LaBSE, ft_gpt_qwen3, ft_gpt_gte, ft_dialemma_bge, ft_gpt_bge`


### Testing Robustness to Dialect Mixing (in Figure 1):

Run `dialect word replacement.ipynb` to generate data with different proportion of mixed dialect. `OpenAi_API_Key` is required. Or use `data/mixed_bar` for convenience.

Run the same evaluation procedure with `script/zs.sh` and `script/eval.sh` to get the results in Figure 1. Change the data root in params to corresponding path.

---
## Acknowledgements

Code in `dense_retrieval.py` and `run_bm25.py` are partly adapted from:

- **Evaluating Large Language Models for Cross-Lingual Information Retrieval** — https://github.com/mainlp/llm-clir

Use of AI:

- AI tools were used during development to assist with debugging, refactoring and drafting small code components.
