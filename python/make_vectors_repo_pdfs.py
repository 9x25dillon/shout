import os, sys, json, re
from pathlib import Path
from typing import List, Dict
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np


CODE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".cpp", ".cc", ".c",
    ".hpp", ".h", ".rb", ".php", ".scala", ".m", ".mm", ".swift", ".cs", ".md", ".txt",
    ".toml", ".yaml", ".yml", ".ini", ".cfg", ".sh", ".bash", ".zsh", ".dockerfile",
    ".gradle", ".mk", ".cmake"
}


def iter_code_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in CODE_EXT:
            try:
                if p.stat().st_size <= 1_000_000:  # 1 MB cap per file for TF-IDF
                    files.append(p)
            except Exception:
                pass
    return files


def read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def chunk_text(t: str, target_tokens: int = 200, overlap: int = 40) -> List[str]:
    words = re.split(r"\s+", t.strip())
    chunks: List[str] = []
    i = 0
    while i < len(words):
        part = words[i : i + target_tokens]
        if len(part) < 30:
            break
        chunks.append(" ".join(part))
        i += max(1, target_tokens - overlap)
    return chunks


def pdf_chunks(pdf_dir: Path) -> List[Dict]:
    out: List[Dict] = []
    for p in pdf_dir.glob("*.pdf"):
        try:
            txt = extract_text(str(p)) or ""
            for j, seg in enumerate(chunk_text(txt)):
                out.append({"id": f"PDF::{p.stem}::p{j+1}", "text": seg})
        except Exception as e:
            print("skip pdf:", p, e)
    return out


def repo_chunks(repo_dir: Path) -> List[Dict]:
    out: List[Dict] = []
    for fp in iter_code_files(repo_dir):
        txt = read_text_file(fp)
        for j, seg in enumerate(chunk_text(txt, target_tokens=220, overlap=60)):
            out.append({"id": f"CODE::{fp.relative_to(repo_dir)}::c{j+1}", "text": seg})
    return out


def embed(chunks: List[Dict], dim: int = 256) -> np.ndarray:
    corpus = [c["text"] for c in chunks]
    vec = TfidfVectorizer(max_features=40000)
    X = vec.fit_transform(corpus)  # (N,F)
    svd = TruncatedSVD(n_components=dim, random_state=0)
    V = svd.fit_transform(X).astype("float32")  # (N,dim)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V


def write_jsonl(path: Path, chunks: List[Dict], V: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ch, vec in zip(chunks, V):
            f.write(json.dumps({"id": ch["id"], "text": ch["text"], "vector": vec.tolist()}) + "\n")


if __name__ == "__main__":
    # usage: python make_vectors_repo_pdfs.py /path/to/repo /path/to/pdfs ./vectors.jsonl
    if len(sys.argv) < 4:
        print("usage: python make_vectors_repo_pdfs.py /path/to/repo /path/to/pdfs ./vectors.jsonl", file=sys.stderr)
        sys.exit(2)
    repo = Path(sys.argv[1])
    pdfs = Path(sys.argv[2])
    out = Path(sys.argv[3])
    chunks = repo_chunks(repo) + pdf_chunks(pdfs)
    assert chunks, "No chunks produced."
    V = embed(chunks, dim=256)
    write_jsonl(out, chunks, V)
    print(f"wrote {len(chunks)} unified vectors to {out}")

