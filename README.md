# 🌸 Maison Elara — AI-Powered Boutique Assistant

A multi-agent AI system for a women's dress boutique, built with LangGraph, CLIP, and FastAPI.

## Architecture

```
User (WebSocket)
      │
      ▼
┌─────────────────────────────────────────┐
│         Main Orchestrator               │
│   ReAct / TAO Loop (LangGraph + Gemini) │
│                                         │
│  Think → Act → Observe → Repeat/Respond │
└──────┬──────┬──────────┬────────────────┘
       │      │          │          │
       ▼      ▼          ▼          ▼
  Stylist  Search     Cart       RAG
  Agent    Agent      Agent      Agent
  (LangGraph) (CLIP   (httpx +  (FAISS +
           + FAISS)   Gemini)   BM25 +
                               CrossEnc.)
```

### Agent Roles

| Agent | Responsibility | Tech |
|-------|---------------|------|
| **Stylist** | Fashion conversation, dress recommendations | LangGraph + Gemini |
| **Search** | Semantic product search by dress description | CLIP + FAISS (singleton) |
| **Cart** | Add/remove items, view cart, place orders | httpx + Gemini tool-use |
| **RAG** | Company policies, returns, shipping FAQ | BM25 + FAISS + CrossEncoder |

### Orchestrator (TAO Pattern)
- **Think**: Gemini decides which agent to call
- **Act**: Calls the appropriate sub-agent tool
- **Observe**: Reads the result, decides if more actions are needed
- **Auto-pipeline**: When stylist gives a suggestion → search is triggered automatically

---

## Project Structure

```
boutique-ai/
├── agents/
│   ├── stylist_agent.py     # LangGraph fashion consultant
│   ├── search_agent.py      # CLIP semantic search (singleton)
│   ├── cart_agent.py        # API cart operations
│   └── rag_agent.py         # Hybrid RAG with reranker
├── orchestrator/
│   └── react_agent.py       # Main TAO orchestrator (per-session)
├── website/
│   └── images/              # Product images (img_0001.png …)
├── company-data/            # Markdown docs for RAG (*.md)
├── frontend/
│   └── index.html           # Chat UI (luxury editorial)
├── server.py                # FastAPI + WebSocket server
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
CART_API_BASE_URL=http://localhost:8000   # URL of your cart/orders API
IMAGE_BASE_PATH=./website/images          # where product images live
FAISS_PRODUCT_INDEX=product_index.faiss   # CLIP product FAISS index
RAG_DOCS_PATH=./company-data              # markdown docs for RAG
```

### 3. Prepare your data

**Product images** (`website/images/`):
```
img_0001.png
img_0002.png
...
```

**CLIP FAISS index** (`product_index.faiss`):
Build it with the provided indexing script (encodes all product images with CLIP).

**Company docs** (`company-data/*.md`):
Add any Markdown files — they'll be auto-indexed by the RAG agent on first run.

### 4. Run the server

```bash
python server.py
```

Visit **http://localhost:8080** in your browser.

---

## Usage Flow

```
User: "I need something for my brother's wedding"
  └─► Orchestrator → Stylist Agent
      └─► "What time of day? What's your style preference?"

User: "Evening, I want to look elegant"
  └─► Orchestrator → Stylist Agent
      └─► {"type": "suggestion", "suggestion": "emerald green satin A-line midi dress..."}
      └─► Orchestrator AUTO-TRIGGERS → Search Agent
          └─► CLIP search → ["0023", "0045", "0012", "0067", "0004"]
          └─► Product cards shown in UI

User: "Add the 2nd one to my cart"
  └─► Orchestrator → Cart Agent
      └─► POST /api/cart/0045
      └─► "Done! Added to your cart 🛍️"

User: "What's your return policy?"
  └─► Orchestrator → RAG Agent
      └─► BM25 + FAISS + CrossEncoder search
      └─► Returns policy info from return-and-refund.md
```

---

## WebSocket Protocol

**Client → Server:**
```json
{ "message": "I need a dress for a wedding" }
```

**Server → Client (event stream):**
```json
{ "type": "status",   "content": "🎨 Consulting your personal stylist…" }
{ "type": "products", "ids": ["0023","0045"], "image_urls": ["/images/img_0023.png", ...], "message": "Found 5 matching products." }
{ "type": "message",  "content": "Here are some options...", "recommended_ids": ["0023","0045",...] }
{ "type": "error",    "content": "Something went wrong" }
```

---

## Building the CLIP Index

```python
import torch, faiss, numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os

device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_dir = "website/images"
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

embeddings = []
ids        = []

for fname in image_files:
    pid = int(fname.replace("img_","").replace(".png",""))
    img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
    inp = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inp)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    embeddings.append(feat.cpu().numpy())
    ids.append(pid)

embeddings = np.vstack(embeddings).astype("float32")
ids_array  = np.array(ids)

index = faiss.IndexIDMap(faiss.IndexFlatIP(512))
index.add_with_ids(embeddings, ids_array)
faiss.write_index(index, "product_index.faiss")
print(f"Indexed {index.ntotal} products.")
```
