"""
Search Agent — CLIP-powered semantic product search.

Loads the model and FAISS index once (singleton pattern) and
provides async-friendly search via asyncio executor.
"""

import asyncio
import os
from functools import lru_cache

from logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# LAZY IMPORTS  (avoid crashing on machines without GPU/CUDA)
# ─────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = os.getenv("FAISS_PRODUCT_INDEX", "product_index.faiss")
IMAGE_BASE_PATH  = os.getenv("IMAGE_BASE_PATH", "website/images/")
TOP_K_DEFAULT    = 5


class SearchAgent:
    """
    Singleton CLIP search agent.

    Usage
    -----
        agent = SearchAgent()
        ids = await agent.search("emerald green satin midi dress", top_k=5)
        # → ["0023", "0045", "0012", "0067", "0004"]
    """

    _instance: "SearchAgent | None" = None

    def __new__(cls) -> "SearchAgent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    # ── Initialization (lazy) ──────────────────────────────

    def _initialize(self) -> None:
        """Load CLIP model + FAISS index (called once, lazily)."""
        if self._ready:
            return

        import torch
        import faiss
        from transformers import CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("[SearchAgent] Loading CLIP model on device=%s", self.device)

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        if not os.path.exists(FAISS_INDEX_PATH):
            log.error("[SearchAgent] FAISS index not found at '%s'", FAISS_INDEX_PATH)
            raise FileNotFoundError(
                f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
                "Please build it first with the indexing script."
            )
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        log.info("[SearchAgent] FAISS index loaded — %d products indexed", self.index.ntotal)

        self._ready = True

    # ── Sync search (runs in thread executor) ─────────────

    def _search_sync(self, query: str, top_k: int) -> list[str]:
        """Blocking CLIP search — call via asyncio executor."""
        import torch

        self._initialize()
        log.debug("[SearchAgent] Encoding query: %r  top_k=%d", query, top_k)

        inputs = self.processor(
            text=[query], return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        query_vector  = text_features.cpu().numpy().astype("float32")

        _, indices = self.index.search(query_vector, top_k)
        results = [f"{i:04d}" for i in indices[0] if i >= 0]
        log.info("[SearchAgent] Search done — query=%r  results=%s", query, results)
        return results

    # ── Async API ──────────────────────────────────────────

    async def search(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[str]:
        """
        Async wrapper around the blocking CLIP search.

        Returns a list of zero-padded product ID strings, e.g. ["0023", "0045"].
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sync, query, top_k)

    # ── Image path helper ──────────────────────────────────

    @staticmethod
    def image_path(product_id: str) -> str:
        """Return the filesystem path for a product image."""
        return os.path.join(IMAGE_BASE_PATH, f"img_{product_id}.png")

    @staticmethod
    def image_url(product_id: str) -> str:
        """Return the URL path served by FastAPI static files."""
        return f"/images/img_{product_id}.png"
