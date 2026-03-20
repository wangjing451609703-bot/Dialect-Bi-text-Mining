import re
from pathlib import Path
import pandas as pd
import hashlib
import shutil

BASE_DIR = Path("/content/drive/MyDrive/data-dass/raw")

# preprocessed csv root
ALIGNED_DIR = BASE_DIR / "aligned_csv"

# eval set outroot
OUT_ROOT = BASE_DIR / "data_eval_fixed99k"

# shared de corpus outroot
CORPUS_SHARED_PATH = OUT_ROOT / "corpus.tsv"

COPY_CORPUS_INTO_EACH_DIALECT_DIR = True

DE_BIG = BASE_DIR / "de.txt"

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def qid_of(dialect: str, text: str) -> str:
    h = hashlib.md5(norm(text).encode("utf-8")).hexdigest()
    return f"{dialect}-{h}"

def read_negatives_after_50k(path: Path, need: int, forbid_set: set[str]):
    out = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(50000):
            if not f.readline():
                break
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            line_n = norm(line)
            if line_n in forbid_set:
                continue
            out.append(line_n)
            if len(out) >= need:
                break
    return out

# out dir
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# read preprocessed csv
de_gsw = pd.read_csv(ALIGNED_DIR / "de-gsw-eval.csv")   # de, gsw
de_swg = pd.read_csv(ALIGNED_DIR / "de-swg-eval.csv")   # de, swg
de_nds = pd.read_csv(ALIGNED_DIR / "de-nds-eval.csv")   # de, nds
de_bar = pd.read_csv(ALIGNED_DIR / "de-bar-eval.csv")   # de, bar
bar_gpt4o = pd.read_csv(BASE_DIR / "de-bar-gpt4o-eval.csv")  # de, bar

# build 1k dialect queries group
gsw_q = list(de_gsw["gsw"].astype(str).tolist()) + list(de_swg["swg"].astype(str).tolist()[:908])
gsw_de = list(de_gsw["de"].astype(str).tolist()) + list(de_swg["de"].astype(str).tolist()[:908])

nds_q = list(de_nds["nds"].astype(str).tolist()[:1000])
nds_de = list(de_nds["de"].astype(str).tolist()[:1000])

bar_q = list(de_bar["bar"].astype(str).tolist()) + list(bar_gpt4o["bar"].astype(str).tolist()[:917])
bar_de = list(de_bar["de"].astype(str).tolist()) + list(bar_gpt4o["de"].astype(str).tolist()[:917])

dialects = {
    "gsw": (gsw_q, gsw_de),
    "nds": (nds_q, nds_de),
    "bar": (bar_q, bar_de),
}

for d, (qs, des) in dialects.items():
    if len(qs) != 1000 or len(des) != 1000:
        raise RuntimeError(f"{d}: expected 1000 queries+1000 pos de, got {len(qs)} / {len(des)}")

# shared positive de set (3k)
all_pos_de = []
for dia in ["gsw", "nds", "bar"]:
    _, des = dialects[dia]
    all_pos_de.extend([norm(x) for x in des])

if len(all_pos_de) != 3000:
    raise RuntimeError(f"Expected 3000 total positives, got {len(all_pos_de)}")

pos_records = []
pos_text_set = set()
for i, de_text in enumerate(all_pos_de, start=1):
    docid = f"POS{i:06d}"    # POS000001..POS003000
    pos_records.append((docid, de_text))
    pos_text_set.add(de_text)

# shared negative de set (97k)
neg_texts = read_negatives_after_50k(DE_BIG, need=97000, forbid_set=pos_text_set)
if len(neg_texts) != 97000:
    raise RuntimeError(f"Need 97000 negatives, but got {len(neg_texts)}. Check de.txt length/content.")

neg_records = [(f"NEG{j:06d}", t) for j, t in enumerate(neg_texts, start=1)]

# shared de corpus 100k
with CORPUS_SHARED_PATH.open("w", encoding="utf-8") as f:
    for docid, text in pos_records:
        f.write(f"{docid}\t{text}\n")
    for docid, text in neg_records:
        f.write(f"{docid}\t{text}\n")

total_corpus = len(pos_records) + len(neg_records)
print(f"[OK] shared corpus -> {CORPUS_SHARED_PATH}")
print(f"     positives={len(pos_records)}, negatives={len(neg_records)}, total={total_corpus}")
assert total_corpus == 100000

# write queries.tsv/qrels.tsv for each dialect
ranges = {
    "gsw": (1, 1000),
    "nds": (1001, 2000),
    "bar": (2001, 3000),
}

for dia in ["gsw", "nds", "bar"]:
    qs, _ = dialects[dia]
    start, end = ranges[dia]

    dia_dir = OUT_ROOT / dia
    dia_dir.mkdir(parents=True, exist_ok=True)

    # queries.tsv
    q_path = dia_dir / "queries.tsv"
    qids = []
    with q_path.open("w", encoding="utf-8") as f:
        for q in qs:
            qn = norm(q)
            qid = qid_of(dia, qn)
            qids.append(qid)
            f.write(f"{qid}\t{qn}\n")

    # qrels.tsv: qid -> POSxxxxxx
    qr_path = dia_dir / "qrels.tsv"
    with qr_path.open("w", encoding="utf-8") as f:
        for idx, qid in enumerate(qids, start=start):
            docid = f"POS{idx:06d}"
            f.write(f"{qid}\t0\t{docid}\t1\n")

    # copy shared corpus to each dialect-de pair
    if COPY_CORPUS_INTO_EACH_DIALECT_DIR:
        shutil.copyfile(CORPUS_SHARED_PATH, dia_dir / "corpus.tsv")

    print(f"[OK] {dia} -> {dia_dir}/queries.tsv, qrels.tsv"
          + (" (+ corpus.tsv copy)" if COPY_CORPUS_INTO_EACH_DIALECT_DIR else ""))

print(f"\nAll done -> {OUT_ROOT}")
