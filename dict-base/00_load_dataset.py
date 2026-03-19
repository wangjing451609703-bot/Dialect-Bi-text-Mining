import argparse
import random
import zipfile
import io
import os
import re
import hashlib
import gzip
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict


VALID_DIALECTS = {"gsw", "nds", "bar"}

def hid(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r"<[^>]+>", " ", s)                 # XML-like
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)   # links
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# sort dialect by file name
def _guess_lang_order_from_name(name: str) -> Tuple[str, str]:
    base = os.path.basename(name).lower()
    base = base.replace(".txt.zip", "").replace(".zip", "").replace(".txt", "")
    m = re.match(r"([a-z]{2,3})-([a-z]{2,3})", base)
    if not m:
        return ("de", "xx")
    l, r = m.group(1), m.group(2)
    if l == "swg": l = "gsw"
    if r == "swg": r = "gsw"
    return (l, r)

def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n\r") for ln in f if ln.strip()]


def _open_text_from_zip(zf: zipfile.ZipFile, info: zipfile.ZipInfo):
    raw = zf.open(info, "r")
    is_gz = info.filename.lower().endswith(".gz")
    return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8", errors="ignore") if is_gz \
           else io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")

def _read_parallel_from_zip(zip_path: Path) -> List[Tuple[str, str]]:

    with zipfile.ZipFile(zip_path, "r") as zf:
        files = [zi for zi in zf.infolist() if not zi.is_dir()]
        lang_exts = {"de", "nds", "gsw", "swg", "bar"}
        by_base = {}
        for fi in files:
            name = fi.filename
            base, dot, ext = name.rpartition(".")
            ext = ext.lower()
            if dot and ext in lang_exts:
                by_base.setdefault(base, {}).update({ext: fi})

        cands = []
        for base, mp in by_base.items():
            if "de" in mp:
                for dia in ("nds", "gsw", "swg", "bar"):
                    if dia in mp:
                        total = mp["de"].file_size + mp[dia].file_size
                        cands.append((total, mp["de"], mp[dia], "gsw" if dia == "swg" else dia))
        if cands:
            _, de_info, dia_info, _dia = max(cands, key=lambda x: x[0])
            with _open_text_from_zip(zf, de_info) as f_de, _open_text_from_zip(zf, dia_info) as f_dia:
                de_lines  = [ln.rstrip("\r\n") for ln in f_de if ln.strip()]
                dia_lines = [ln.rstrip("\r\n") for ln in f_dia if ln.strip()]
            if len(de_lines) != len(dia_lines):
                raise RuntimeError(f"Line mismatch in zip pair: {de_info.filename} vs {dia_info.filename}")
            return list(zip(de_lines, dia_lines))

        
        def is_text(n: str) -> bool:
            n = n.lower()
            return n.endswith((".txt", ".tsv", ".csv", ".txt.gz", ".tsv.gz", ".csv.gz"))
        texts = [fi for fi in files if is_text(fi.filename)]
        target = max((texts or files), key=lambda x: x.file_size)

        
        with _open_text_from_zip(zf, target) as fh:
            peek = []
            for _ in range(50):
                s = fh.readline()
                if not s: break
                s = s.rstrip("\r\n")
                if s: peek.append(s)
        with _open_text_from_zip(zf, target) as fh:
            def detect_sep(samples):
                def ok(cnts): return sum(c >= 1 for c in cnts) >= max(1, len(cnts)//3)
                tabs   = [s.count("\t") for s in samples]
                pipes  = [s.count(" ||| ") for s in samples]
                commas = [s.count(",") for s in samples]
                if ok(tabs):   return "\t"
                if ok(pipes):  return " ||| "
                if ok(commas): return ","
                return None
            sep = detect_sep(peek)
            if sep is None:
                return []
            pairs = []
            for line in fh:
                line = line.rstrip("\r\n")
                if not line: continue
                parts = line.split(sep)
                if len(parts) < 2: continue
                a, b = parts[0].strip(), parts[1].strip()
                if a and b:
                    pairs.append((a, b))
            return pairs

def read_wikimatrix_pair_any(p: Path) -> Tuple[List[str], List[str], str]:

    if p.is_file() and p.suffix == ".zip":
        pairs = _read_parallel_from_zip(p)  # [(de, dia)] 或 [(left, right)]
        if not pairs:
            return [], [], "xx"
        L, R = _guess_lang_order_from_name(p.name)
        if L == "de" and R != "de":
            de_texts  = [a for (a, b) in pairs]
            dia_texts = [b for (a, b) in pairs]
            code = R
        elif R == "de" and L != "de":
            de_texts  = [b for (a, b) in pairs]
            dia_texts = [a for (a, b) in pairs]
            code = L
        else:
            de_texts  = [a for (a, b) in pairs]
            dia_texts = [b for (a, b) in pairs]
            code = "xx"
        if code == "swg": code = "gsw"
        return dia_texts, de_texts, code


    if p.is_dir():
        cand_de = sorted(list(p.glob("*.de")))
        if not cand_de:
            return [], [], "xx"
        de_file = cand_de[0]
        base = de_file.name[:-3]
        dia_file, code = None, None
        for L in ("nds", "gsw", "swg", "bar"):
            other = p / f"{base}{L}"
            if other.exists():
                dia_file = other; code = "gsw" if L == "swg" else L; break
        if dia_file is None:
            return [], [], "xx"
    else:
        if p.name.endswith(".de"):
            de_file = p
            base = p.name[:-3]
            dia_file, code = None, None
            for L in ("nds", "gsw", "swg", "bar"):
                cand = p.with_name(base + L)
                if cand.exists():
                    dia_file = cand; code = "gsw" if L == "swg" else L; break
            if dia_file is None:
                return [], [], "xx"
        else:
            dia_file = p
            if p.name.endswith(".swg"):
                code = "gsw"
            elif p.suffix in (".nds", ".gsw", ".bar"):
                code = p.suffix[1:]
            else:
                return [], [], "xx"
            base = p.name[: -len(p.suffix)]
            de_file = p.with_name(base + ".de")
            if not de_file.exists():
                return [], [], "xx"

    de_lines  = _read_lines(de_file)
    dia_lines = _read_lines(dia_file)
    if len(de_lines) != len(dia_lines):
        return [], [], "xx"
    if code == "swg": code = "gsw"
    return dia_lines, de_lines, code

# exculede eval data from training corpus
def _load_eval_exclusions(eval_root: Path):

    excl_queries, excl_docs = set(), set()
    for dia in ("gsw", "nds", "bar"):
        qf = eval_root / dia / "queries.tsv"
        cf = eval_root / dia / "corpus.tsv"
        if qf.exists():
            with qf.open("r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    sp = ln.rstrip("\n").split("\t", 1)
                    if len(sp) == 2 and sp[1]:
                        excl_queries.add(_norm(sp[1]))   # <- normalization
        if cf.exists():
            with cf.open("r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    sp = ln.rstrip("\n").split("\t", 1)
                    if len(sp) == 2 and sp[1]:
                        excl_docs.add(_norm(sp[1]))      # <-  normalization
    return excl_queries, excl_docs

def _gather_all_pairs(sources: List[Path]):
    buckets = defaultdict(list)
    seen = set()
    for p in sources:
        dia_texts, de_texts, code = read_wikimatrix_pair_any(p)
        code = (code or "").lower()
        if code == "swg": code = "gsw"
        if code not in VALID_DIALECTS:
            continue
        for dx, det in zip(dia_texts, de_texts):
            dx = _norm(dx); det = _norm(det)
            if len(dx) < 2 or len(det) < 2:
                continue
            key = (code, dx, det)
            if key in seen:
                continue
            seen.add(key)
            buckets[code].append((dx, det))
    return buckets

def _filter_out_eval_leak(buckets: dict, excl_queries: set, excl_docs: set):
    filtered = {}
    removed_total = 0
    for code, pairs in buckets.items():
        before = len(pairs)
        kept = [(q, d) for (q, d) in pairs if (q not in excl_queries and d not in excl_docs)]
        filtered[code] = kept
        removed = before - len(kept)
        removed_total += removed
        print(f"[EXCLUDE-APPLIED] {code}: removed={removed} kept={len(kept)}")
    print(f"[EXCLUDE-APPLIED] total removed={removed_total}")
    return filtered

def _split_train_dev_by_dialect(buckets: dict, dev_ratio: float, seed: int):
    rng = random.Random(seed)
    train, dev = [], []
    for code, pairs in buckets.items():
        pairs = pairs[:]
        rng.shuffle(pairs)
        n = len(pairs)
        if n == 0:
            continue
        n_dev = max(1, int(n * dev_ratio))
        dev.extend([(q, d, code) for (q, d) in pairs[:n_dev]])
        train.extend([(q, d, code) for (q, d) in pairs[n_dev:]])
        print(f"[SPLIT] {code}: total={n} dev={n_dev} train={n - n_dev}")
    print(f"[TOTAL] train={len(train)} dev={len(dev)}")
    return train, dev

def _write_ir_triplet(pairs, outdir: Path, prefix: str = ""):
    outdir.mkdir(parents=True, exist_ok=True)
    doc2id = {}
    qid = 0

    corpus = {}
    with (outdir/"queries.tsv").open("w", encoding="utf-8") as fq, \
         (outdir/"qrels.tsv").open("w", encoding="utf-8") as fr:
        for qtxt, dtxt, _code in pairs:
            if dtxt not in doc2id:
                did = "D" + hashlib.md5(dtxt.encode("utf-8")).hexdigest()[:16]
                doc2id[dtxt] = did
                corpus[did] = dtxt
            did = doc2id[dtxt]
            qid_str = f"Q{prefix}{qid:09d}"
            qid += 1
            fq.write(f"{qid_str}\t{qtxt}\n")
            fr.write(f"{qid_str}\t{did}\t1\n")
    with (outdir/"corpus.tsv").open("w", encoding="utf-8") as fc:
        for did, txt in corpus.items():
            fc.write(f"{did}\t{txt}\n")
    print(f"[WRITE] {outdir}: corpus={len(corpus)} queries={qid}")

def main():
    ap = argparse.ArgumentParser(description="raw zip to fine-tune- / synthetize- ready dataset")
    ap.add_argument("--out_dir", required=True, help="output repository")
    ap.add_argument("--eval_root", required=True, help="eval set root repository")
    ap.add_argument("--dev_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gsw_zips", nargs="+", required=True)
    ap.add_argument("--nds_zips", nargs="+", required=True)
    ap.add_argument("--bar_zips", nargs="+", required=True)

    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) load all dialect data
    sources = [Path(p) for p in (args.gsw_zips + args.nds_zips + args.bar_zips)]

    # 2) align and exclude duplicated
    buckets = _gather_all_pairs(sources)
    for code in VALID_DIALECTS:
        print(f"[LOAD] {code}: {len(buckets.get(code, []))} pairs before exclusion")

    # 3) exclude eval data
    excl_q, excl_d = _load_eval_exclusions(Path(args.eval_root))
    print(f"[EXCLUDE] eval queries={len(excl_q)} docs={len(excl_d)}")
    buckets = _filter_out_eval_leak(buckets, excl_q, excl_d)
    for code in VALID_DIALECTS:
        print(f"[KEEP] {code}: {len(buckets.get(code, []))} pairs after exclusion")

    # 4) split to train/dev
    train_pairs, dev_pairs = _split_train_dev_by_dialect(buckets, args.dev_ratio, args.seed)

    # 5) write tsv
    _write_ir_triplet(train_pairs, out_root/"train", prefix="T")
    _write_ir_triplet(dev_pairs,   out_root/"dev",   prefix="D")

    print(f"[OK] Wrote IR triplets to {out_root}/train and {out_root}/dev")

if __name__ == "__main__":
    main()
