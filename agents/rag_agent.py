"""
RAG Agent — Hybrid BM25 + FAISS retrieval with CrossEncoder reranking.

Answers questions about company policies, shipping, returns, etc.

Expected folder layout (relative to where you run the server):
  ./company-data/
      return-and-refund.md
      payment-and-checkout.md
      shipping.md
      ...any other *.md files

Indexes are cached after first build:
  ./faiss_vector_store_index/   ← FAISS index directory
  ./bm25.pkl                    ← BM25 pickle
"""

import glob
import os
import pickle
from typing import List, Tuple

import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings   # replaces deprecated community class
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ─────────────────────────────────────────────────────────────
# CONFIG  — override any of these via environment variables
# ─────────────────────────────────────────────────────────────

DOCS_PATH       = os.getenv("RAG_DOCS_PATH",   "./company-data")
FAISS_PATH      = os.getenv("RAG_FAISS_PATH",  "./faiss_vector_store_index")
BM25_PATH       = os.getenv("RAG_BM25_PATH",   "./bm25.pkl")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL  = os.getenv("RERANKER_MODEL",  "BAAI/bge-reranker-base")


# ─────────────────────────────────────────────────────────────
# DOCUMENT LOADING
# ─────────────────────────────────────────────────────────────

def load_markdown_docs() -> List[Document]:
    """
    Load and chunk all *.md files from DOCS_PATH.
    Splits on Markdown headers so each chunk has meaningful context.
    Raises ValueError if no documents are found.
    """
    md_files = glob.glob(f"{DOCS_PATH}/*.md")

    # ── Guard: tell the user exactly what's wrong ────────────
    if not md_files:
        abs_path = os.path.abspath(DOCS_PATH)
        raise FileNotFoundError(
            f"\n\n[RAGAgent] No .md files found in '{abs_path}'.\n"
            f"  • Make sure your company docs are saved as Markdown files (*.md)\n"
            f"  • and placed in the '{DOCS_PATH}' folder next to server.py.\n"
            f"  • Current working directory: {os.getcwd()}\n"
        )

    docs = []
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = splitter.split_text(text)

        # ── If the file has no headers, treat the whole file as one chunk ──
        if not chunks:
            docs.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(file_path), "full_path": file_path},
            ))
            continue

        for chunk in chunks:
            if not chunk.page_content.strip():
                continue
            metadata = chunk.metadata or {}
            metadata.update({"source": os.path.basename(file_path), "full_path": file_path})
            docs.append(Document(page_content=chunk.page_content, metadata=metadata))

    if not docs:
        raise ValueError(
            f"[RAGAgent] Found {len(md_files)} .md file(s) in '{DOCS_PATH}' "
            f"but all chunks were empty. Check that your files contain text."
        )

    print(f"[RAGAgent] Loaded {len(docs)} chunks from {len(md_files)} file(s) in '{DOCS_PATH}'")
    return docs


# ─────────────────────────────────────────────────────────────
# BM25 RETRIEVER
# ─────────────────────────────────────────────────────────────

class BM25Retriever:
    def __init__(self, documents: List[Document]):
        if not documents:
            raise ValueError("[BM25Retriever] Cannot build index from empty document list.")
        self.docs   = documents
        self.corpus = [doc.page_content.split() for doc in documents if doc.page_content.strip()]
        if not self.corpus:
            raise ValueError("[BM25Retriever] All documents have empty content.")
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        scores    = self.bm25.get_scores(query.split())
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [(self.docs[i], float(scores[i])) for i in top_k_idx]


def _load_or_create_bm25(docs: List[Document]) -> BM25Retriever:
    """
    Load BM25 from disk if it exists, otherwise build and save it.
    Note: `docs` must always be provided (not None) — caller is responsible.
    """
    if os.path.exists(BM25_PATH):
        print(f"[RAGAgent] Loading BM25 index from disk: {BM25_PATH}")
        with open(BM25_PATH, "rb") as f:
            return pickle.load(f)
    print("[RAGAgent] Building new BM25 index…")
    bm25 = BM25Retriever(docs)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"[RAGAgent] BM25 index saved to {BM25_PATH}")
    return bm25


# ─────────────────────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────────────────────

def _load_or_create_vectorstore(docs: List[Document]) -> FAISS:
    """
    Load FAISS from disk if it exists, otherwise build and save it.
    Note: `docs` must always be provided (not None) — caller is responsible.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(FAISS_PATH):
        print(f"[RAGAgent] Loading FAISS vector store from disk: {FAISS_PATH}")
        return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("[RAGAgent] Building new FAISS vector store…")
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(FAISS_PATH)
    print(f"[RAGAgent] FAISS vector store saved to {FAISS_PATH}")
    return vs


# ─────────────────────────────────────────────────────────────
# HYBRID RETRIEVER
# ─────────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(self, vectorstore: FAISS, bm25: BM25Retriever):
        self.vectorstore = vectorstore
        self.bm25        = bm25

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k)
        bm25_results   = self.bm25.retrieve(query, k=k)

        bm25_scores = [s for _, s in bm25_results]
        bm25_max    = max(bm25_scores) if bm25_scores else 1.0

        combined: dict[str, Tuple[Document, float]] = {}

        for doc, dist in vector_results:
            combined[doc.page_content] = (doc, 1 / (1 + dist))

        for doc, score in bm25_results:
            key        = doc.page_content
            norm_score = score / bm25_max if bm25_max > 0 else 0.0
            if key in combined:
                combined[key] = (doc, combined[key][1] + norm_score)
            else:
                combined[key] = (doc, norm_score)

        return list(combined.values())


# ─────────────────────────────────────────────────────────────
# CROSS-ENCODER RERANKER
# ─────────────────────────────────────────────────────────────

class Reranker:
    def __init__(self):
        print(f"[RAGAgent] Loading reranker model: {RERANKER_MODEL}…")
        self.model = CrossEncoder(RERANKER_MODEL)
        print("[RAGAgent] Reranker ready.")

    def rerank(
        self,
        query: str,
        docs: List[Tuple[Document, float]],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        pairs  = [(query, doc.page_content) for doc, _ in docs]
        scores = self.model.predict(pairs)
        scored = [(doc, float(s)) for (doc, _), s in zip(docs, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ─────────────────────────────────────────────────────────────
# RAG AGENT  (singleton)
# ─────────────────────────────────────────────────────────────

class RAGAgent:
    """
    Singleton RAG agent. Heavy models (embeddings, reranker) are loaded
    once on first query and reused for all subsequent calls.

    Usage
    -----
        agent = RAGAgent()
        results = await agent.query("What if quality check fails?", top_k=3)
    """

    _instance: "RAGAgent | None" = None

    def __new__(cls) -> "RAGAgent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def _initialize(self) -> None:
        if self._ready:
            return

        print("[RAGAgent] Initializing…")

        # ── FIX: check each index independently ─────────────
        # The old code used a single `indexes_exist` flag for BOTH indexes,
        # which meant if FAISS existed but BM25 didn't, docs=None was passed
        # to the BM25 builder → empty corpus → ZeroDivisionError.
        faiss_exists = os.path.exists(FAISS_PATH)
        bm25_exists  = os.path.exists(BM25_PATH)

        # Only load docs if at least one index needs to be built
        docs: List[Document] | None = None
        if not faiss_exists or not bm25_exists:
            docs = load_markdown_docs()   # raises clearly if folder is empty

        # Build/load each index independently
        self.vectorstore = _load_or_create_vectorstore(docs if not faiss_exists else docs or self._load_docs_fallback())
        self.bm25        = _load_or_create_bm25(docs if not bm25_exists else docs or self._load_docs_fallback())

        self.hybrid   = HybridRetriever(self.vectorstore, self.bm25)
        self.reranker = Reranker()
        self._ready   = True
        print("[RAGAgent] Ready.")

    @staticmethod
    def _load_docs_fallback() -> List[Document]:
        """Called only when one index exists and the other doesn't — reloads docs."""
        return load_markdown_docs()

    async def query(self, user_query: str, top_k: int = 5) -> List[dict]:
        """
        Returns top_k relevant chunks, reranked by CrossEncoder.
        Each result: {"content": str, "score": float, "source": str}
        """
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._initialize)

        def _retrieve_rerank():
            retrieved = self.hybrid.retrieve(user_query, k=10)
            reranked  = self.reranker.rerank(user_query, retrieved, top_k=top_k)
            return reranked

        reranked = await loop.run_in_executor(None, _retrieve_rerank)

        return [
            {
                "content": doc.page_content,
                "score":   score,
                "source":  doc.metadata.get("source", "unknown"),
            }
            for doc, score in reranked
        ]