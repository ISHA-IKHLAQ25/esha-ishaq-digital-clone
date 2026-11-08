"""Retrieval half of Esha's digital clone.

Pipeline: load markdown notes -> chunk (~250 words, overlap) -> BM25 index + dense
embeddings -> rank with BM25, dense cosine, or Reciprocal Rank Fusion of both.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHUNKS_PATH = ROOT / "chunks.json"
EMBEDDINGS_PATH = ROOT / "embeddings.npy"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 250      # target size of a chunk
OVERLAP_WORDS = 40     # words carried over from the previous chunk
RRF_K = 60             # constant in score = sum 1 / (RRF_K + rank)

MODES = ("Hybrid (RRF)", "BM25 only", "Dense only")


# ---------------------------------------------------------------- chunking
@dataclass
class Chunk:
    chunk_id: str   # e.g. "projects-03"
    source: str     # file name, e.g. "projects.md"
    section: str    # nearest markdown heading
    text: str

    @property
    def indexed_text(self) -> str:
        """Text used by both retrievers: the heading gives short chunks some context."""
        return f"{self.section}\n{self.text}"


def _blocks(markdown: str):
    """Yield (heading, paragraph) pairs; paragraphs are blank-line separated."""
    heading = ""
    for block in re.split(r"\n\s*\n", markdown):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            first, _, rest = block.partition("\n")
            heading = first.lstrip("#").strip()
            block = rest.strip()
            if not block:
                continue
        yield heading, block


def chunk_document(path: Path) -> list[Chunk]:
    """Pack paragraphs into ~CHUNK_WORDS-word chunks, overlapping by OVERLAP_WORDS."""
    chunks: list[Chunk] = []
    words: list[str] = []
    section = None  # heading of the first new (non-overlap) paragraph in the current chunk

    def flush(carry: bool) -> None:
        nonlocal words, section
        if not words:
            return
        cid = f"{path.stem}-{len(chunks):02d}"
        chunks.append(Chunk(cid, path.name, section or "", " ".join(words)))
        words = words[-OVERLAP_WORDS:] if carry else []
        section = None

    for heading, para in _blocks(path.read_text(encoding="utf-8")):
        para_words = para.split()
        if words and section is not None and len(words) + len(para_words) > CHUNK_WORDS:
            flush(carry=True)
        if section is None:
            section = heading
        # a single very long paragraph is cut into word windows
        while len(para_words) > CHUNK_WORDS:
            room = CHUNK_WORDS - len(words)
            words += para_words[:room]
            para_words = para_words[room:]
            flush(carry=True)
        words += para_words
    flush(carry=False)
    return chunks


def load_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")) + sorted(data_dir.glob("*.txt")):
        chunks.extend(chunk_document(path))
    return chunks


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.indexed_text.encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- BM25
STOPWORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being
do does did done have has had i me my mine you your yours we our it its this that these those
what which who whom whose when where why how there here so than then too very can could would
should will shall may might must about into over under again just also not no yes any some
""".split())


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # tiny plural stemmer: projects -> project
        out.append(t)
    return out


# ---------------------------------------------------------------- retriever
@dataclass
class Hit:
    chunk: Chunk
    score: float                 # score in the mode that produced the hit
    bm25_score: float
    dense_score: float
    bm25_rank: int               # 1-based rank in the BM25 list
    dense_rank: int              # 1-based rank in the dense list

    def as_dict(self) -> dict:
        d = asdict(self)
        d["chunk"] = asdict(self.chunk)
        return d


class Retriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings are out of sync; run build_index.py")
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks])

    # raw scores over the whole corpus
    def bm25_scores(self, question: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(question)), dtype=np.float32)

    def dense_scores(self, question: str) -> np.ndarray:
        q = self.embedder.encode(question, normalize_embeddings=True, convert_to_numpy=True)
        return self.embeddings @ q.astype(np.float32)  # cosine: both sides are unit vectors

    @staticmethod
    def _ranks(scores: np.ndarray) -> np.ndarray:
        """1-based rank of every chunk (rank 1 = best)."""
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=int)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

    def search(self, question: str, mode: str = MODES[0], k: int = 4) -> list[Hit]:
        bm25 = self.bm25_scores(question)
        dense = self.dense_scores(question)
        bm25_rank, dense_rank = self._ranks(bm25), self._ranks(dense)

        if mode == "BM25 only":
            final = bm25
        elif mode == "Dense only":
            final = dense
        else:  # Reciprocal Rank Fusion: score = sum over lists of 1 / (60 + rank)
            # A chunk with BM25 score 0 shares no word with the question, so it gets no
            # BM25 vote (otherwise its tie-broken "rank" would leak into the fusion).
            bm25_vote = np.where(bm25 > 0, 1.0 / (RRF_K + bm25_rank), 0.0)
            final = bm25_vote + 1.0 / (RRF_K + dense_rank)

        top = np.argsort(-final, kind="stable")[:k]
        return [
            Hit(self.chunks[i], float(final[i]), float(bm25[i]), float(dense[i]),
                int(bm25_rank[i]), int(dense_rank[i]))
            for i in top
        ]


# ---------------------------------------------------------------- build / load
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed_chunks(chunks: list[Chunk], embedder) -> np.ndarray:
    return embedder.encode([c.indexed_text for c in chunks], normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def save_index(chunks: list[Chunk], embeddings: np.ndarray) -> None:
    np.save(EMBEDDINGS_PATH, embeddings)
    CHUNKS_PATH.write_text(json.dumps({
        "fingerprint": corpus_fingerprint(chunks),
        "embed_model": EMBED_MODEL,
        "chunks": [asdict(c) for c in chunks],
    }, indent=1, ensure_ascii=False), encoding="utf-8")


def load_retriever() -> Retriever:
    """Load precomputed embeddings; re-embed only if the data files changed."""
    chunks = load_chunks()
    embedder = load_embedder()
    saved = json.loads(CHUNKS_PATH.read_text(encoding="utf-8")) if CHUNKS_PATH.exists() else {}
    if EMBEDDINGS_PATH.exists() and saved.get("fingerprint") == corpus_fingerprint(chunks):
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        print("[rag] data changed since embeddings.npy was built; re-embedding", flush=True)
        embeddings = embed_chunks(chunks, embedder)
    return Retriever(chunks, embeddings, embedder)
