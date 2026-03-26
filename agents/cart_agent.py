"""
Cart Agent — Handles all cart & order operations via API calls.

Wraps the LLM-powered CheckoutAgent from the original code.
Exposes a clean async `run(message, recommended_ids, history)` interface.
"""

import json
import os
from typing import Any

import httpx
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BASE_URL = os.getenv("CART_API_BASE_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def format_cart_summary(cart_items: list[dict]) -> str:
    """Deterministic formatter — avoids LLM hallucination on prices."""
    if not cart_items:
        return "Your cart is empty."
    lines = []
    total = 0
    for item in cart_items:
        lines.append(
            f"- {item['name']}: {item['quantity']} items | "
            f"Price: {item['price']} | Subtotal: {item['subtotal']}"
        )
        total += item["subtotal"]
    return "Cart Summary:\n" + "\n".join(lines) + f"\n\nTotal: {total}"


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def build_system_prompt(recommended_ids: list[str]) -> str:
    product_map = "\n".join(
        f'  - Position {i+1} ("{ordinal(i+1)}"): product_id = "{pid}"'
        for i, pid in enumerate(recommended_ids)
    )

    return f"""
You are a checkout assistant for a women's dress boutique.

Recommended products (current search results):
{product_map if product_map else "  (no products searched yet)"}

-------------------------
STRICT RULES
-------------------------
- Use ONLY data returned by tools.
- NEVER modify, round, or convert prices.
- NEVER estimate totals.
- Always display prices EXACTLY as returned.
- If "computed_summary" is present → use it EXACTLY.

-------------------------
BEHAVIOR
-------------------------
- Resolve "1st", "2nd", "first", "second", etc. → the correct product_id
- Call tools when needed
- Keep responses short and helpful

DO NOT hallucinate product details or pricing.
""".strip()


# ─────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────

@tool
async def add_to_cart(product_id: str) -> str:
    """Add a product to the cart using its product ID."""
    log.info("[Cart] add_to_cart product_id=%s", product_id)
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(f"{BASE_URL}/api/cart/{product_id}")
            r.raise_for_status()
            log.debug("[Cart] add_to_cart success: %s", r.json())
            return _json_text(r.json())
        except httpx.HTTPStatusError as e:
            log.error("[Cart] add_to_cart HTTP error: %s", e.response.text)
            return _json_text({"error": e.response.text})
        except Exception as e:
            log.error("[Cart] add_to_cart error: %s", e, exc_info=True)
            return _json_text({"error": str(e)})


@tool
async def remove_from_cart(product_id: str) -> str:
    """Remove a product from the cart using its product ID."""
    log.info("[Cart] remove_from_cart product_id=%s", product_id)
    async with httpx.AsyncClient() as client:
        try:
            r = await client.delete(f"{BASE_URL}/api/cart/{product_id}")
            r.raise_for_status()
            log.debug("[Cart] remove_from_cart success: %s", r.json())
            return _json_text(r.json())
        except httpx.HTTPStatusError as e:
            log.error("[Cart] remove_from_cart HTTP error: %s", e.response.text)
            return _json_text({"error": e.response.text})
        except Exception as e:
            log.error("[Cart] remove_from_cart error: %s", e, exc_info=True)
            return _json_text({"error": str(e)})


@tool
async def view_cart() -> str:
    """Retrieve all items currently in the cart."""
    log.info("[Cart] view_cart called")
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{BASE_URL}/api/cart")
            r.raise_for_status()
            items = r.json()
            log.debug("[Cart] view_cart returned %d items", len(items))
            return _json_text({"cart_items": items})
        except httpx.HTTPStatusError as e:
            log.error("[Cart] view_cart HTTP error: %s", e.response.text)
            return _json_text({"error": e.response.text})
        except Exception as e:
            log.error("[Cart] view_cart error: %s", e, exc_info=True)
            return _json_text({"error": str(e)})


@tool
async def place_order() -> str:
    """Place an order for all items currently in the cart."""
    log.info("[Cart] place_order called")
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(f"{BASE_URL}/api/orders")
            r.raise_for_status()
            log.info("[Cart] place_order success: %s", r.json())
            return _json_text(r.json())
        except httpx.HTTPStatusError as e:
            log.error("[Cart] place_order HTTP error: %s", e.response.text)
            return _json_text({"error": e.response.text})
        except Exception as e:
            log.error("[Cart] place_order error: %s", e, exc_info=True)
            return _json_text({"error": str(e)})


# ─────────────────────────────────────────────────────────────
# CART AGENT CLASS
# ─────────────────────────────────────────────────────────────

class CartAgent:
    """
    LLM-powered cart agent that resolves ordinal references
    (1st, 2nd …) to real product IDs and calls the cart API.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    async def run(
        self,
        user_message: str,
        recommended_ids: list[str],
        conversation_history: list[BaseMessage] | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        user_message : str
            Raw user request (e.g. "add the 2nd item to cart").
        recommended_ids : list[str]
            Product IDs from the latest search result.
        conversation_history : optional
            Prior cart conversation messages (for multi-turn cart dialogues).

        Returns
        -------
        dict  {"reply": str, "actions": list[dict]}
        """
        tools = [add_to_cart, remove_from_cart, view_cart, place_order]
        tools_by_name = {t.name: t for t in tools}
        model = self.llm.bind_tools(tools)

        messages: list[BaseMessage] = [
            SystemMessage(content=build_system_prompt(recommended_ids))
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append(HumanMessage(content=user_message))

        actions_taken: list[dict] = []

        while True:
            ai_msg = await model.ainvoke(messages)
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", []) or []

            # ─ Final answer ─
            if not tool_calls:
                reply = (ai_msg.content or "").strip()
                log.info("[CartAgent] Final reply (%d chars): %r", len(reply), reply[:100])
                return {
                    "reply":   reply,
                    "actions": actions_taken,
                }

            # ─ Execute tools ─
            for call in tool_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})
                tool_fn   = tools_by_name[tool_name]
                log.info("[CartAgent] Calling tool=%s  args=%s", tool_name, tool_args)

                result = await tool_fn.ainvoke(tool_args)
                log.debug("[CartAgent] Tool result: %s", result[:200] if isinstance(result, str) else result)

                # ─ Special: deterministic cart display ─
                if tool_name == "view_cart":
                    parsed = json.loads(result)
                    if "cart_items" in parsed:
                        summary = format_cart_summary(parsed["cart_items"])
                        return {
                            "reply":   summary,
                            "actions": actions_taken,
                        }

                actions_taken.append(
                    {"tool": tool_name, "args": tool_args, "result": result}
                )
                messages.append(
                    ToolMessage(content=result, tool_call_id=call["id"])
                )
