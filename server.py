"""
Maison Elara — unified server (single process, single port).

All traffic goes through :8080.  Vite dev-server proxies /api, /images, /ws.
In production, FastAPI also serves the React build via the SPA catch-all.

URL map
───────
WS   /ws                     AI stylist WebSocket (one MainAgent per conn)
GET  /api/products            paginated product list
GET  /api/products/search     CLIP semantic search (shared singleton, 0 double-load)
GET  /api/products/{pid}      single product
GET  /api/cart                cart contents with subtotals
POST /api/cart/{pid}          add / increment
DEL  /api/cart/{pid}          remove entirely
PUT  /api/cart/{pid}          set explicit quantity
POST /api/orders              place order (empties cart, decrements stock)
GET  /api/orders              order history
GET  /images/{filename}       product images (StaticFiles)
GET  /health                  health probe
GET  /                        React SPA (or Vite dev-server in dev mode)
GET  /{any}                   SPA catch-all for client-side routes
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, Integer, String, asc, create_engine, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

from logger import get_logger
log = get_logger(__name__)

# ── AI agents (import after load_dotenv so env vars are available) ────────────
from orchestrator.react_agent import MainAgent
from agents.search_agent import SearchAgent

# Singleton: CLIP is loaded once here, reused by both the REST endpoint
# and the AI agent's internal tool calls.
_search_agent = SearchAgent()


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./shop.db")
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class Product(Base):
    __tablename__ = "products"
    id         = Column(String,  primary_key=True)
    name       = Column(String,  nullable=False)
    brand      = Column(String,  nullable=False)
    price      = Column(Float,   nullable=False)
    rating     = Column(Integer, nullable=False)
    quantity   = Column(Integer, default=20)
    image_path = Column(String,  nullable=False)


class CartItem(Base):
    __tablename__ = "cart"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String,  nullable=False, unique=True)
    quantity   = Column(Integer, default=1)
    added_at   = Column(DateTime, default=datetime.now)


class Order(Base):
    __tablename__ = "orders"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    product_id    = Column(String,  nullable=False)
    product_name  = Column(String)
    product_brand = Column(String)
    product_image = Column(String)
    quantity      = Column(Integer, nullable=False)
    unit_price    = Column(Float,   nullable=False)
    total_price   = Column(Float,   nullable=False)
    ordered_at    = Column(DateTime, default=datetime.now)
    status        = Column(String,  default="placed")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def p_dict(p: Product) -> dict:
    return {
        "id": p.id, "name": p.name, "brand": p.brand,
        "price": p.price, "rating": p.rating,
        "quantity": p.quantity, "image_path": p.image_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class UpdateQty(BaseModel):
    quantity: int


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Maison Elara", version="2.0.0", docs_url="/api/docs")

# CORS — allow all origins so Vite dev-server (port 3000) can reach port 8080
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static mounts (must come before route definitions) ────────────────────────

IMAGE_DIR = os.getenv("IMAGE_BASE_PATH", "./website/images")
if os.path.isdir(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")
else:
    log.warning("[server] IMAGE_BASE_PATH '%s' not found — /images will 404", IMAGE_DIR)

# React production build assets (JS/CSS bundles live at /static/*)
REACT_BUILD = Path(os.getenv("REACT_BUILD_DIR", "./frontend/dist"))
REACT_INDEX = REACT_BUILD / "index.html"
if (REACT_BUILD / "static").is_dir():
    app.mount("/static", StaticFiles(directory=str(REACT_BUILD / "static")), name="react-static")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Maison Elara"}


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/products")
def list_products(
    page:  int = 1,
    limit: int = 20,
    sort:  str = "default",
    db:    Session = Depends(get_db),
):
    """Paginated product list. sort: default | price_asc | price_desc | rating"""
    q = db.query(Product)
    sort_map = {
        "price_asc":  asc(Product.price),
        "price_desc": desc(Product.price),
        "rating":     desc(Product.rating),
    }
    q     = q.order_by(sort_map.get(sort, Product.id))
    total = q.count()
    items = q.offset((page - 1) * limit).limit(limit).all()
    return {
        "total": total,
        "page":  page,
        "pages": max(1, math.ceil(total / limit)),
        "items": [p_dict(p) for p in items],
    }


@app.get("/api/products/search")
async def search_products_endpoint(q: str, top_k: int = 20, db: Session = Depends(get_db)):
    """
    CLIP text-to-image semantic search.
    Shares the SearchAgent singleton with the AI agent — CLIP loaded once.
    """
    log.info("[/api/products/search] q=%r top_k=%d", q, top_k)
    ids = await _search_agent.search(q, top_k=top_k)
    if not ids:
        return {"total": 0, "items": []}
    rows    = db.query(Product).filter(Product.id.in_(ids)).all()
    by_id   = {p.id: p for p in rows}
    # preserve CLIP ranking order
    ordered = [p_dict(by_id[i]) for i in ids if i in by_id]
    log.info("[/api/products/search] returning %d results", len(ordered))
    return {"total": len(ordered), "items": ordered}


@app.get("/api/products/{pid}")
def get_product(pid: str, db: Session = Depends(get_db)):
    p = db.query(Product).filter(Product.id == pid).first()
    if not p:
        raise HTTPException(404, "Product not found")
    return p_dict(p)


# ─────────────────────────────────────────────────────────────────────────────
# CART
# These are the endpoints CartAgent calls via httpx → CART_API_BASE_URL.
# With the unified server both live on :8080, so no cross-process calls.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/cart")
def get_cart(db: Session = Depends(get_db)):
    items  = db.query(CartItem).order_by(CartItem.added_at.desc()).all()
    result = []
    for item in items:
        p = db.query(Product).filter(Product.id == item.product_id).first()
        if p:
            result.append({
                "cart_id":    item.id,
                "product_id": item.product_id,
                "quantity":   item.quantity,
                "name":       p.name,
                "brand":      p.brand,
                "price":      p.price,
                "image_path": p.image_path,
                "subtotal":   round(item.quantity * p.price, 2),
                "added_at":   item.added_at.isoformat(),
            })
    return result


@app.post("/api/cart/{pid}")
def add_to_cart(pid: str, db: Session = Depends(get_db)):
    """Add 1 unit; increments quantity if already in cart. No request body."""
    if not db.query(Product).filter(Product.id == pid).first():
        raise HTTPException(404, f"Product '{pid}' not found")
    existing = db.query(CartItem).filter(CartItem.product_id == pid).first()
    if existing:
        existing.quantity += 1
        db.commit()
        return {"message": "Cart updated", "cart_id": existing.id, "quantity": existing.quantity}
    ci = CartItem(product_id=pid, quantity=1)
    db.add(ci); db.commit(); db.refresh(ci)
    return {"message": "Added to cart", "cart_id": ci.id, "quantity": 1}


@app.delete("/api/cart/{pid}")
def remove_from_cart(pid: str, db: Session = Depends(get_db)):
    item = db.query(CartItem).filter(CartItem.product_id == pid).first()
    if not item:
        raise HTTPException(404, "Item not in cart")
    db.delete(item); db.commit()
    return {"message": "Removed from cart"}


@app.put("/api/cart/{pid}")
def update_cart_qty(pid: str, body: UpdateQty, db: Session = Depends(get_db)):
    """Set explicit quantity. quantity ≤ 0 removes the item."""
    item = db.query(CartItem).filter(CartItem.product_id == pid).first()
    if not item:
        raise HTTPException(404, "Item not in cart")
    if body.quantity <= 0:
        db.delete(item)
    else:
        item.quantity = body.quantity
    db.commit()
    return {"message": "Cart updated"}


# ─────────────────────────────────────────────────────────────────────────────
# ORDERS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/orders")
def place_order(db: Session = Depends(get_db)):
    """Place order for entire cart. Empties cart and decrements stock."""
    cart = db.query(CartItem).all()
    if not cart:
        raise HTTPException(400, "Cart is empty")
    placed = []
    for item in cart:
        p = db.query(Product).filter(Product.id == item.product_id).first()
        if not p:
            continue
        db.add(Order(
            product_id    = p.id,
            product_name  = p.name,
            product_brand = p.brand,
            product_image = p.image_path,
            quantity      = item.quantity,
            unit_price    = p.price,
            total_price   = round(item.quantity * p.price, 2),
        ))
        p.quantity = max(0, p.quantity - item.quantity)
        placed.append(p.id)
    db.query(CartItem).delete()
    db.commit()
    log.info("[/api/orders] Placed order for %d products", len(placed))
    return {"success": True, "placed": placed, "count": len(placed)}


@app.get("/api/orders")
def get_orders(db: Session = Depends(get_db)):
    orders = db.query(Order).order_by(Order.ordered_at.desc()).all()
    return [
        {
            "id":            o.id,
            "product_id":    o.product_id,
            "product_name":  o.product_name,
            "product_brand": o.product_brand,
            "product_image": o.product_image,
            "quantity":      o.quantity,
            "unit_price":    o.unit_price,
            "total_price":   o.total_price,
            "ordered_at":    o.ordered_at.isoformat(),
            "status":        o.status,
        }
        for o in orders
    ]


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — AI AGENT
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent  = MainAgent()
    client = websocket.client
    log.info("[WS] New session — client=%s:%s", client.host, client.port)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                user_message: str = json.loads(raw).get("message", "").strip()
            except json.JSONDecodeError:
                user_message = raw.strip()

            if not user_message:
                continue

            log.info("[WS] message=%r", user_message)
            try:
                async for event in agent.stream_run(user_message):
                    log.debug("[WS] event type=%s", event.get("type"))
                    await websocket.send_json(event)
            except Exception as exc:
                log.error("[WS] Agent error: %s", exc, exc_info=True)
                await websocket.send_json({"type": "error", "content": str(exc)})

    except WebSocketDisconnect:
        log.info("[WS] Session closed — client=%s:%s", client.host, client.port)
    except Exception as exc:
        log.error("[WS] Unexpected error: %s", exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# REACT SPA — must be the very last route
# In development Vite handles the frontend; this only matters for production.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_spa(full_path: str):
    # Serve real build files (favicon, manifest, etc.) directly
    candidate = REACT_BUILD / full_path
    if candidate.is_file():
        return FileResponse(str(candidate))
    # All other paths → index.html (React Router takes over)
    if REACT_INDEX.exists():
        return FileResponse(str(REACT_INDEX))
    # Dev mode: Vite is serving the frontend, FastAPI shouldn't interfere
    return HTMLResponse(
        "<h1>Maison Elara API is running.</h1>"
        "<p>In development, open <a href='http://localhost:3000'>localhost:3000</a> "
        "for the React frontend.</p>"
        "<p><a href='/api/docs'>API docs →</a></p>",
        status_code=200,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        # ⚠️  Keep reload=False in production.
        # reload=True causes uvicorn to fork a subprocess which re-imports
        # everything — CLIP would be loaded twice and on GPU you'd OOM.
        reload=False,
        log_level="info",
    )