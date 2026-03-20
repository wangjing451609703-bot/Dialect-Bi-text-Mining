import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/content/drive/MyDrive/data-dass/raw")

def read_lines(p: Path):
    lines = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\r\n")
            if ln.strip():
                lines.append(ln)
    return lines

def find_pair(prefix_candidates):
    """
    find (de_file, dialect_file, dialect_code) in BASE_DIR
    prefix_candidates: [(base, dia_ext)], e.g. [("Tatoeba.de-swg", "swg")]
    """
    for base, dia_ext in prefix_candidates:
        de_file = BASE_DIR / f"{base}.de"
        dia_file = BASE_DIR / f"{base}.{dia_ext}"
        if de_file.exists() and dia_file.exists():
            return de_file, dia_file, dia_ext
    return None, None, None

def make_csv(de_file: Path, dia_file: Path, dia_code: str, out_path: Path):
    de_lines = read_lines(de_file)
    dia_lines = read_lines(dia_file)

    if len(de_lines) != len(dia_lines):
        raise RuntimeError(
            f"Line count mismatch!\n"
            f"  de : {de_file.name} -> {len(de_lines)} lines\n"
            f"  dia: {dia_file.name} -> {len(dia_lines)} lines"
        )

    df = pd.DataFrame({"de": de_lines, dia_code: dia_lines})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] {dia_code}: {len(df)} pairs -> {out_path}")
    display(df.head(3))
    return out_path

tasks = [
    # swg
    ("swg", [("Tatoeba.de-swg", "swg")], "de-swg-eval.csv"),
    # gsw
    ("gsw", [("Tatoeba.de-gsw", "gsw")], "de-gsw-eval.csv"),
    # bar: 兼容 Tatoeba.bar-de.* 或 Tatoeba.de-bar.*
    ("bar", [("Tatoeba.bar-de", "bar"), ("Tatoeba.de-bar", "bar")], "de-bar-eval.csv"),
    # nds
    ("nds", [("Tatoeba.de-nds", "nds")], "de-nds-eval.csv"),
]


out_dir = BASE_DIR / "aligned_csv"
out_paths = []

for dia_code, candidates, out_name in tasks:
    de_file, dia_file, dia_ext = find_pair(candidates)
    if de_file is None:
        raise FileNotFoundError(
            f"Cannot find files for dialect={dia_code}. Tried bases: {candidates}\n"
            f"Make sure the required files exist in: {BASE_DIR}"
        )
    out_paths.append(
        make_csv(de_file, dia_file, dia_ext, out_dir / out_name)
    )

print("\nAll done. Generated files:")
for p in out_paths:
    print(" -", p)
