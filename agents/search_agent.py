"""
SearchAgent — CLIP + FAISS semantic product search (singleton).

Supports two FAISS index formats automatically:
  1. IndexIDMap  — indices[0] contains the actual product IDs directly.
  2. IndexFlatIP — indices[0] contains row numbers; needs image_ids.pkl to map
                   row → product ID. This is what main.py used.

On first search the agent detects which format the index uses and picks the
right path, so you do not need to change anything when switching index types.
"""

import asyncio
import os
import pickle

FAISS_INDEX_PATH = os.getenv("FAISS_PRODUCT_INDEX", "product_index.faiss")
IDS_PKL_PATH     = os.getenv("IMAGE_IDS_PKL",       "image_ids.pkl")
IMAGE_BASE_PATH  = os.getenv("IMAGE_BASE_PATH",     "website/images/")
TOP_K_DEFAULT    = 5


class SearchAgent:
    _instance: "SearchAgent | None" = None

    def __new__(cls) -> "SearchAgent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    # ── lazy init ────────────────────────────────────────────

    def _initialize(self) -> None:
        if self._ready:
            return

        import faiss
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SearchAgent] Loading CLIP on {self.device}…")
        self.model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        print("[SearchAgent] CLIP ready.")

        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"[SearchAgent] FAISS index not found: '{FAISS_INDEX_PATH}'\n"
                "Copy product_index.faiss from your Jupyter folder."
            )
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        # Detect index type: IndexIDMap stores IDs internally; plain IndexFlatIP needs pkl.
        index_type = type(self.index).__name__
        self._use_id_map = "IDMap" in index_type

        # Load pkl if present — used for plain IndexFlatIP
        self._image_ids: list[int] | None = None
        if not self._use_id_map:
            if not os.path.exists(IDS_PKL_PATH):
                raise FileNotFoundError(
                    f"[SearchAgent] index type is '{index_type}' (needs row→ID mapping) "
                    f"but '{IDS_PKL_PATH}' was not found.\n"
                    "Run in Jupyter:  pickle.dump(image_ids, open('image_ids.pkl','wb'))"
                )
            with open(IDS_PKL_PATH, "rb") as f:
                self._image_ids = pickle.load(f)
            print(f"[SearchAgent] Loaded image_ids.pkl — {len(self._image_ids)} IDs")

        print(f"[SearchAgent] Index ready — {self.index.ntotal} vectors | mode={'IDMap' if self._use_id_map else 'pkl'}")
        self._ready = True

    # ── sync search (run in thread executor) ─────────────────

    def _search_sync(self, query: str, top_k: int) -> list[str]:
        import torch

        self._initialize()

        inputs = self.processor(
            text=[query], return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            feat = self.model.get_text_features(**inputs)
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        qv   = feat.cpu().numpy().astype("float32")

        _, raw_indices = self.index.search(qv, top_k)

        ids: list[str] = []
        for val in raw_indices[0]:
            if val < 0:          # FAISS returns -1 for empty slots
                continue
            if self._use_id_map:
                # val IS the product ID stored in the index
                ids.append(f"{int(val):04d}")
            else:
                # val is a row number; map through pkl list
                row = int(val)
                if 0 <= row < len(self._image_ids):
                    ids.append(f"{self._image_ids[row]:04d}")
        return ids

    # ── public async API ──────────────────────────────────────

    async def search(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[str]:
        """Returns zero-padded product ID strings: ["0023", "0045", ...]"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sync, query, top_k)

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def image_url(product_id: str) -> str:
        return f"/images/img_{product_id}.png"

    @staticmethod
    def image_path(product_id: str) -> str:
        return os.path.join(IMAGE_BASE_PATH, f"img_{product_id}.png")