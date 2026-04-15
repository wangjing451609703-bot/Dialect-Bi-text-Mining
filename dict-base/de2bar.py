"""
Replace de-word to bar and generate Bavarian-like queries.
Use DiaLemma to enable word-by-word substitution and build qrel map to match query-document pairs.
Set a max expansion to 30 as 1 de-word might have multiple corresponding bar-variants.
"""

import argparse, json, itertools, random
from pathlib import Path
from typing import Dict, List, Tuple


def read_tsv_pairs(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            rows.append((parts[0], parts[1]))
    return rows

def write_tsv_pairs(path: Path, rows: List[Tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k, v in rows:
            f.write(f"{k}\t{v}\n")

def build_doc_to_qid_map(qrels_path: Path) -> Dict[str, str]:
    mp: Dict[str,str] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            qid, docid, rel = parts[0], parts[1], parts[2]
            if docid not in mp:
                mp.setdefault(docid, []).append(qid)
    return mp

def pick_one_qid(docid: str, qids: List[str]) -> str:
    if not qids:
       return ""
    return qids[0]

def load_dict_from_dialemma(jsonl_path: Path, lowercase: bool) -> Dict[str, List[str]]:
    mp: Dict[str, List[str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                line = line.rstrip(",")
                obj = json.loads(line)
            de = obj.get("term")
            if not isinstance(de, str) or not de.strip():
                continue
            cands: List[str] = []
            dts = obj.get("translations") or []
            if isinstance(dts, list):
                for dt in dts:
                    if isinstance(dt, str) and dt.strip():
                        cands.append(dt.strip())
            vs = obj.get("inflected_variants") or []
            if isinstance(vs, list):
                for v in vs:
                    if isinstance(v, str) and v.strip():
                        cands.append(v.strip())
            if not cands:
                continue
            key = de.strip()
            if lowercase:
                key = key.lower()
            seen = set()
            clean = []
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    clean.append(c)
            if key in mp:
                for c in clean:
                    if c not in mp[key]:
                        mp[key].append(c)
            else:
                mp[key] = clean
    return mp

def expand_replacements(tokens: List[str], repl_map: Dict[str, List[str]], lowercase: bool,
                        max_expansions: int) -> List[List[str]]:
    slots: List[List[str]] = []
    num_replace_positions = 0
    for t in tokens:
        k = t.lower() if lowercase else t
        if k in repl_map:
            num_replace_positions += 1
            slots.append(repl_map[k])
        else:
            slots.append([t])
    if num_replace_positions == 0:
        return []
    out = []
    for combo in itertools.product(*slots):
        out.append(list(combo))
        if len(out) >= max_expansions:
            break
    return out

def process_split(data_split_dir: Path, out_split_dir: Path,
                  dialemma_path: Path,
                  lowercase_match: bool, max_expansions: int, add_variant_suffix: bool):
    corpus_path = data_split_dir / "corpus.tsv"
    qrels_path  = data_split_dir / "qrels.tsv"
    if not corpus_path.exists() or not qrels_path.exists():
        print(f"[SKIP] missing files in {data_split_dir}")
        return

    doc2qids = build_doc_to_qid_map(qrels_path)

    dicts = load_dict_from_dialemma(dialemma_path, lowercase_match)

    corpus_rows = read_tsv_pairs(corpus_path)

    out_rows: List[Tuple[str,str]] = []
    skipped_nohit = 0
    skipped_noqid = 0
    total_variants = 0

    for docid, detext in corpus_rows:
        qids = doc2qids.get(docid,[])
        if not qids:
            skipped_noqid += 1
            continue
        qid = pick_one_qid(docid, qids)

        toks = detext.strip().split()
        variants = expand_replacements(toks, dicts, lowercase_match, max_expansions)
        if not variants:
            skipped_nohit += 1
            continue

        for idx, toks_out in enumerate(variants):
            out_qid = qid
            if add_variant_suffix and len(variants) > 1:
                out_qid = f"{qid}-v{idx+1}"
            out_rows.append((out_qid, " ".join(toks_out)))
        total_variants += len(variants)

    out_path = out_split_dir / f"bar_dict_trans_queries.tsv"
    write_tsv_pairs(out_path, out_rows)
    print(f"[{data_split_dir.name}] wrote {len(out_rows)} rows; total_variants={total_variants}; skipped(no hit)={skipped_nohit}; missing_qid={skipped_noqid}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root with train/ or dev/ (corpus.tsv, qrels.tsv)")
    ap.add_argument("--out_root", required=True, help="Output root for Step 3 results")
    ap.add_argument("--dialemma_path", required=True)
    ap.add_argument("--lowercase_match", action="store_true")
    ap.add_argument("--max_expansions", type=int, default=30, help="Cap number of expanded variants per sentence")
    ap.add_argument("--no_variant_suffix", action="store_true", help="Do not append -v{idx} when multiple variants are produced")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    dialemma_path = Path(args.dialemma_path)


    splits = []
    for name in ("train","dev"):
        sdir = data_root / name
        if sdir.exists():
            splits.append(sdir)
    if not splits:
        print(f"[ERR] No split directories under {data_root}")
        return

    for sdir in splits:
        out_split = out_root / sdir.name
        out_split.mkdir(parents=True, exist_ok=True)
        process_split(sdir, out_split, dialemma_path,
                      args.lowercase_match, args.max_expansions, not args.no_variant_suffix)

    print(f"[OK] Done. Outputs: {out_root}/<split>/bar_dict_trans_queries.tsv")

if __name__ == "__main__":
    main()
