import argparse
import gzip
import hashlib
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Set


VALID_DIALECTS = {"gsw", "nds", "bar"}

def hid(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r"<[^>]+>", " ", s)                 # XML-like
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)   # links
    s = re.sub(r"\s+", " ", s)
    return s.strip()

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

# load corpus
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

# align wiki pairs line by line, return in (dialect_texts, de_texts, dialect_code)
def read_wikimatrix_pair_any(p: Path) -> Tuple[List[str], List[str], str]:

    if p.is_file() and p.suffix == ".zip":
        pairs = _read_parallel_from_zip(p) 
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

# read/write tsv
def read_tsv_pairs(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            p = line.split("\t", 1)
            if len(p) != 2:
                continue
            rows.append((p[0], p[1]))
    return rows

def write_tsv_pairs(path: Path, rows: List[Tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a, b in rows:
            f.write(f"{a}\t{b}\n")

def write_corpus_dup(out_dir: Path, corpus_rows: List[Tuple[str,str]]):
    for d in ("nds","bar","gsw"):
        write_tsv_pairs(out_dir / f"{d}_corpus.tsv", corpus_rows)

# bag of sentences
def build_bags(zip_paths: List[Path], match_side: str) -> Dict[str, Set[str]]:

    bags: Dict[str, Set[str]] = {d:set() for d in VALID_DIALECTS}
    for zp in zip_paths:
        dia_texts, de_texts, code = read_wikimatrix_pair_any(zp)
        if code not in VALID_DIALECTS:
            continue
        if match_side == "dialect":
            for s in dia_texts:
                bags[code].add(_norm(s))
        elif match_side == "de":
            for s in de_texts:
                bags[code].add(_norm(s))
        else:
            raise ValueError("match_side must be 'dialect' or 'de'")
    return bags

def classify_queries(queries: List[Tuple[str,str]], bags: Dict[str,Set[str]], on_unknown: str) -> Dict[str, List[Tuple[str,str]]]:
    buckets: Dict[str, List[Tuple[str,str]]] = {d: [] for d in ("gsw","nds","bar")}
    unknown: List[Tuple[str,str]] = []
    for qid, qtext in queries:
        qt = _norm(qtext)
        hit = None
        for d in ("gsw","nds","bar"):
            if qt in bags.get(d, set()):
                hit = d
                break
        if hit:
            buckets[hit].append((qid, qtext))
        else:
            if on_unknown == "assign_gsw":
                buckets["gsw"].append((qid, qtext))
            elif on_unknown == "keep":
                unknown.append((qid, qtext))
            elif on_unknown == "drop":
                pass
    if on_unknown == "keep" and unknown:
        buckets["unknown"] = unknown
    return buckets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root with train/ or dev/ (corpus.tsv, queries.tsv, qrels.tsv)")
    ap.add_argument("--out_root", required=True, help="Output root")
    ap.add_argument("--zips", nargs="+", required=True, help="Zip or directory/files of wiki corpus")
    ap.add_argument("--match_side", choices=["dialect","de","auto"], default="auto", help="Build bags from which side")
    ap.add_argument("--on_unknown", choices=["assign_gsw","keep","drop"], default="keep")
    ap.add_argument("--dump_unknown", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    zip_paths = [Path(z) for z in args.zips]

    # build sentence bags on both side
    dial_bags = build_bags(zip_paths, "dialect")
    de_bags   = build_bags(zip_paths, "de")

    for split in ("train","dev"):
        sdir = data_root / split
        if not sdir.exists():
            continue
        corpus = read_tsv_pairs(sdir / "corpus.tsv")
        queries = read_tsv_pairs(sdir / "queries.tsv")

        out_split = out_root / split
        out_split.mkdir(parents=True, exist_ok=True)
        write_corpus_dup(out_split, corpus)

        # choose match_side
        if args.match_side == "auto":
            # check coverage
            qn = [ _norm(qt) for _, qt in queries ]
            dial_cov = sum(1 for qt in qn if any(qt in dial_bags[d] for d in VALID_DIALECTS))
            de_cov   = sum(1 for qt in qn if any(qt in de_bags[d]   for d in VALID_DIALECTS))
            match_side = "dialect" if dial_cov >= de_cov else "de"
            print(f"[{split}] auto-select match_side={match_side} (dialect_cov={dial_cov}, de_cov={de_cov}, total={len(queries)})")
            bags = dial_bags if match_side == "dialect" else de_bags
        else:
            match_side = args.match_side
            bags = dial_bags if match_side == "dialect" else de_bags
            print(f"[{split}] match_side={match_side} (total={len(queries)})")

        buckets = classify_queries(queries, bags, args.on_unknown)

        # write tsv
        for d in ("gsw","nds","bar"):
            write_tsv_pairs(out_split / f"{d}_queries.tsv", buckets.get(d, []))
        if "unknown" in buckets:
            write_tsv_pairs(out_split / "unknown_queries.tsv", buckets["unknown"])
            if args.dump_unknown > 0:
                dump = buckets["unknown"][: args.dump_unknown]
                write_tsv_pairs(out_split / "unknown_samples.tsv", dump)

        total_in = len(queries)
        total_out = sum(len(buckets.get(d, [])) for d in ("gsw","nds","bar")) + (len(buckets.get("unknown", [])) if args.on_unknown=="keep" else 0)
        print(f"[{split}] in={total_in} -> gsw:{len(buckets.get('gsw',[]))} nds:{len(buckets.get('nds',[]))} bar:{len(buckets.get('bar',[]))}"
              + (f" unknown:{len(buckets.get('unknown',[]))}" if args.on_unknown=='keep' else '')
              + f" | out_total={total_out}")

    print("[OK] split-by-matching v4 completed.")

if __name__ == "__main__":
    main()