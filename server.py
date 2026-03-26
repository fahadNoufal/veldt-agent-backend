"""
FastAPI server for the Maison Elara boutique AI assistant.

Endpoints
---------
GET  /                    → serves frontend/index.html
GET  /images/{filename}   → serves product images (static)
WS   /ws                  → WebSocket chat (one agent per connection)
GET  /health              → health check
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from logger import get_logger
from orchestrator.react_agent import MainAgent

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="Maison Elara — Boutique AI", version="1.0.0")

# Serve product images
IMAGE_DIR = os.getenv("IMAGE_BASE_PATH", "./website/images")
if os.path.isdir(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH))
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Maison Elara AI"}


# ─────────────────────────────────────────────────────────────
# WEBSOCKET  — one MainAgent instance per connection
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Fresh agent = fresh session state (stylist history, recommended_ids)
    agent = MainAgent()
    client = websocket.client
    log.info("[WS] New session opened — client=%s:%s", client.host, client.port)

    try:
        while True:
            # ── Receive user message ──────────────────────
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                user_message: str = data.get("message", "").strip()
            except json.JSONDecodeError:
                user_message = raw.strip()

            if not user_message:
                continue

            log.info("[WS] User message: %r", user_message)

            # ── Stream agent responses ────────────────────
            try:
                async for event in agent.stream_run(user_message):
                    log.debug("[WS] Sending event type=%s", event.get("type"))
                    await websocket.send_json(event)
            except Exception as exc:
                log.error("[WS] Agent error: %s", exc, exc_info=True)
                await websocket.send_json({
                    "type":    "error",
                    "content": f"Something went wrong: {str(exc)}",
                })

    except WebSocketDisconnect:
        log.info("[WS] Session closed — client=%s:%s", client.host, client.port)
    except Exception as exc:
        log.error("[WS] Unexpected error: %s", exc, exc_info=True)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
