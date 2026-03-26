"""
Main Orchestrator — LangGraph ReAct Agent (TAO loop + Chain-of-Thought).

Architecture
────────────
  create_react_agent(llm, tools, prompt)
      └─ LangGraph prebuilt ReAct graph
         ├─ Node: "agent"  — LLM reasons & decides next action
         └─ Node: "tools"  — executes the chosen tool
         (loops until LLM stops calling tools)

  Streaming: agent.astream({"messages": [...]}, stream_mode="updates")
      yields per-node diffs so the frontend gets live status updates
      as each tool is called, not just at the end.

Session state (per WebSocket connection)
────────────────────────────────────────
  chat_history     — full orchestrator-level message log (LangChain messages)
                     passed into every invocation so the agent reads context
  stylist_history  — serialized stylist sub-agent turns (separate memory)
  recommended_ids  — product IDs from the latest CLIP search result
"""

import json
from typing import AsyncIterator, Any

from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from agents.stylist_agent import StylistAgent
from agents.search_agent  import SearchAgent
from agents.cart_agent    import CartAgent
from agents.rag_agent     import RAGAgent
from logger import get_thinking_logger

_think_log = get_thinking_logger()


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT  — detailed CoT / TAO instructions
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are ELARA, the master AI coordinator for "Maison Elara" — a premium women's dress boutique.
Your role is to deliver a seamless, warm, and intelligent shopping experience by orchestrating
4 specialized sub-agents. You have access to the full conversation history on every turn.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING PROCESS  (Chain-of-Thought + TAO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before EVERY response, reason silently through these steps:

  THOUGHT  → What does the user actually need right now?
             Read the conversation history carefully.
             What has already been asked/answered?
             What is the user's current shopping stage?

  ACTION   → Which tool should I call? With what exact input?
             Am I missing any context (e.g., no suggestion yet, need to ask stylist first)?

  OBSERVE  → What did the tool return?
             Is the result a "question" or "suggestion"?
             Do I need to chain another tool call?
             Is the user's request fully resolved?

  REPEAT   → If not resolved, loop back to THOUGHT with the new observation.

  RESPOND  → Only when the user's intent is fully addressed.
             Synthesize all observations into one warm, coherent reply.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE 4 TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ask_stylist(user_message: str)
   ─────────────────────────────
   Purpose : Conversational fashion consultant. Understands occasion, style,
             time-of-day, and fit preferences to recommend a specific dress.
   Returns : JSON  {"type": "question", "questions": "..."}
          OR JSON  {"type": "suggestion", "suggestion": "...", "message": "...",
                    "follow_up": "...", "user_preference": "..."}
   When    : Use for ANY fashion/style/occasion conversation.
             Pass the user's exact message — include all context.

2. search_products(dress_description: str)
   ───────────────────────────────────────
   Purpose : CLIP semantic search over the product catalog using a visual
             dress description. Returns ranked product IDs and image URLs.
   Returns : JSON {"product_ids": [...], "image_urls": [...], "count": N}
   When    : Call IMMEDIATELY and AUTOMATICALLY whenever ask_stylist returns
             {"type": "suggestion"}. Extract the "suggestion" field verbatim
             and pass it as dress_description. Do NOT wait for user to ask.
   Never   : Call this without a prior suggestion. Never make up a description.

3. handle_cart(user_message: str)
   ────────────────────────────────
   Purpose : Manages the shopping cart and orders via API calls.
             Resolves ordinal references ("1st", "2nd", "third"…) to the
             product IDs from the most recent search automatically.
   Actions : add item, remove item, view cart, place order, checkout
   Returns : Confirmation text or cart summary.
   When    : Whenever the user mentions cart, add, remove, buy, order,
             checkout, purchase, place order, or refers to numbered items
             (e.g. "add the 2nd one").
   Pass    : The user's EXACT original message — do not paraphrase.

4. ask_rag(question: str)
   ──────────────────────
   Purpose : Retrieves answers from the company knowledge base using hybrid
             BM25+vector search with cross-encoder reranking.
   Covers  : Returns policy, refund process, quality checks, shipping times,
             delivery tracking, payment methods, exchange policy, sizing info,
             store hours, contact details, gift wrapping, promotions.
   Returns : Ranked document chunks with source and score.
   When    : Any question about company policy, shipping, payments, or
             store information.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION STAGE AWARENESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read the chat history to determine the user's current stage:

  STAGE 0 — No context yet
    User just arrived. No occasion, style, or preference known.
    → Route to ask_stylist to start discovery.

  STAGE 1 — Gathering preferences
    Stylist has asked clarifying questions. User is answering.
    → Continue routing to ask_stylist with the user's answers.
    → Do NOT call search_products yet — wait for a "suggestion" type response.

  STAGE 2 — Suggestion made, no search yet
    Stylist returned {"type": "suggestion"} in this turn or a prior turn,
    but search_products has NOT been called yet.
    → Call search_products immediately with the suggestion text.

  STAGE 3 — Search done, products shown
    Product IDs exist in session. User is browsing.
    → If user says "add the 2nd", "I like the first one", "buy it" → handle_cart.
    → If user says "show me something else" or "not quite right" → ask_stylist again.
    → If user asks "how do I return?" → ask_rag.

  STAGE 4 — Cart / order
    User is managing their cart or placing an order.
    → handle_cart for all actions.
    → ask_rag if they ask about payment, delivery, or policies mid-checkout.

  MIXED queries (e.g., "add to cart and tell me the return policy"):
    → Chain multiple tools in sequence: handle_cart THEN ask_rag.
    → Combine the results into one coherent response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL CHAINING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ask_stylist → SUGGESTION → search_products   (always chain these two)
  handle_cart + ask_rag                         (can be called in same turn)
  ask_stylist → QUESTION → stop, relay question (do NOT chain search yet)
  ask_rag alone                                 (policy questions stand alone)

  Maximum TAO iterations per turn: 6
  After 6 loops without a final answer, summarise what you know and respond.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STYLE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  After a stylist QUESTION:
    → Relay the question naturally, as if you are asking it yourself.
    → Warm, conversational tone. One or two questions max.

  After a stylist SUGGESTION + search:
    → Lead with the stylist's "message".
    → Say: "We found [N] styles matching that description — see them above!"
    → Include the stylist's "follow_up".
    → Guide: "You can say 'add the 1st item', 'add the 3rd one', etc."

  After a cart action:
    → Short, warm confirmation. E.g. "Done! I've added that to your cart. 🛍️"
    → If cart is shown: present the summary cleanly.

  After a RAG answer:
    → Answer directly from the retrieved content.
    → Cite the source document name naturally (e.g. "According to our returns policy…").
    → Keep it concise — 2–4 sentences unless more detail is needed.

  After a mixed query (multiple tools):
    → Address each part of the request in order.
    → Use a natural transition between topics.

  Tone: warm, elegant, knowledgeable. Like a personal shopper at a luxury boutique.
  Length: as short as possible while being complete. Avoid filler phrases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✗ NEVER invent product details, prices, dimensions, or availability.
  ✗ NEVER make up company policies, shipping times, or refund rules.
  ✗ NEVER call search_products without a stylist suggestion.
  ✗ NEVER answer policy questions from memory — always use ask_rag.
  ✗ NEVER skip the search step after a suggestion — always chain it.
  ✓ ALWAYS read the chat history before deciding which tool to call.
  ✓ ALWAYS pass the user's original message to handle_cart verbatim.
  ✓ ALWAYS relay the stylist's exact questions/messages to the user.
""".strip()


# ─────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────

def _msg_text(msg: Any) -> str:
    """Safely extract text content from any message or raw value."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _extract_ai_text(msg: AIMessage) -> str:
    """
    Extract the visible text from a Gemini AIMessage, handling both:
      - Plain string content  (older models / non-thinking mode)
      - List-of-parts content (Gemini 2.5 Flash with thinking enabled)

    Gemini 2.5 Flash returns content as a list like:
      [
        {"type": "thinking", "thinking": "<internal reasoning>"},
        {"type": "text",     "text":     "<final answer>"},
      ]

    Thinking parts are logged to logs/thinking.log and stripped from
    the value returned to the frontend.
    """
    content = msg.content

    # Simple string — no thinking involved
    if isinstance(content, str):
        return content.strip()

    # List of content parts (Gemini 2.5 Flash)
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "thinking":
                thinking_text = part.get("thinking", "")
                if thinking_text:
                    _think_log.debug("[Thinking]\n%s", thinking_text)
            elif ptype == "text":
                t = part.get("text", "").strip()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts).strip()

    # Fallback
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


# ─────────────────────────────────────────────────────────────
# PER-SESSION MAIN AGENT
# ─────────────────────────────────────────────────────────────

class MainAgent:
    """
    One instance per WebSocket session.

    State
    ─────
    chat_history    : full LangChain message log passed to every agent invocation.
                      Gives the LLM full context so it reasons about stage correctly.
    stylist_history : serialized sub-agent memory (separate from orchestrator).
    recommended_ids : product IDs from the latest search, used by cart agent.
    """

    def __init__(self):
        # ── Sub-agents ─────────────────────────────────────
        self.stylist = StylistAgent()
        self.search  = SearchAgent()   # singleton — CLIP model loaded once globally
        self.cart    = CartAgent()
        self.rag     = RAGAgent()      # singleton — FAISS + BM25 loaded once globally

        # ── Session state ──────────────────────────────────
        self.chat_history:    list[BaseMessage] = []
        self.stylist_history: list[dict]        = []
        self.recommended_ids: list[str]         = []

        # ── LLM ───────────────────────────────────────────
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

    # ─────────────────────────────────────────────────────
    # TOOL FACTORY
    # Each call to _build_tools() returns fresh @tool closures
    # that capture `self`, so they always read the latest
    # session state (recommended_ids, stylist_history, etc.)
    # ─────────────────────────────────────────────────────

    def _build_tools(self) -> list:
        session = self  # closure capture

        @tool
        async def ask_stylist(user_message: str) -> str:
            """
            Consult the fashion stylist sub-agent for dress recommendations.
            Maintains its own conversation history across turns.

            Input  : The user's message (pass verbatim for best results).
            Output : JSON object — either:
                     {"type": "question", "questions": "<clarifying question>"}
                  OR {"type": "suggestion", "suggestion": "<dress description>",
                      "message": "<..>", "follow_up": "<..>", "user_preference": "<..>"}

            If the return type is "suggestion", you MUST immediately call
            search_products with the "suggestion" field as input.
            """
            result = await session.stylist.chat(user_message, session.stylist_history)
            session.stylist_history = result["history"]
            return json.dumps(result["response"], ensure_ascii=False)

        @tool
        async def search_products(dress_description: str) -> str:
            """
            Search the product catalog using CLIP visual-semantic embeddings.
            Call this ONLY after ask_stylist returns {"type": "suggestion"}.

            Input  : The exact "suggestion" text from the stylist output.
                     Example: "emerald green satin A-line midi dress with pleats"
            Output : JSON with product_ids list, image_urls list, and count.
                     Product IDs are zero-padded strings: "0001", "0023", etc.
                     Users can reference these by position: "1st", "2nd", etc.
            """
            ids = await session.search.search(dress_description, top_k=5)
            session.recommended_ids = ids
            image_urls = [SearchAgent.image_url(pid) for pid in ids]
            return json.dumps({
                "product_ids": ids,
                "image_urls":  image_urls,
                "count":       len(ids),
            }, ensure_ascii=False)

        @tool
        async def handle_cart(user_message: str) -> str:
            """
            Manage the shopping cart and orders via the store API.
            Resolves "1st", "2nd", "third" etc. to real product IDs automatically
            using the results from the most recent search_products call.

            Supports: add item, remove item, view cart, place order, checkout.

            Input  : The user's EXACT original message — do not rephrase.
                     Examples: "add the 2nd item to my cart"
                               "remove the first one"
                               "show me my cart"
                               "place my order"
            Output : Confirmation text or formatted cart summary.
            """
            result = await session.cart.run(
                user_message=user_message,
                recommended_ids=session.recommended_ids,
            )
            return json.dumps(result, ensure_ascii=False)

        @tool
        async def ask_rag(question: str) -> str:
            """
            Query the company knowledge base for policy and store information.
            Uses hybrid BM25 + vector retrieval with cross-encoder reranking.

            Covers: return policy, refund process, quality check outcomes,
                    shipping times, delivery tracking, payment methods,
                    exchange policy, sizing charts, store hours, promotions,
                    gift wrapping, contact details.

            Input  : A natural language question about company or store info.
            Output : Ranked document excerpts with source file and relevance score.
                     Always answer from these results — never from memory.
            """
            results = await session.rag.query(question, top_k=3)
            if not results:
                return "No relevant information found in the knowledge base."
            return "\n\n".join(
                f"[Source: {r['source']} | Relevance: {r['score']:.2f}]\n{r['content']}"
                for r in results
            )

        return [ask_stylist, search_products, handle_cart, ask_rag]

    # ─────────────────────────────────────────────────────
    # BUILD REACT GRAPH
    # create_react_agent builds a LangGraph graph:
    #   SystemMessage injected at the top
    #   Node "agent"  → LLM reasons and issues tool calls
    #   Node "tools"  → executes tool, appends ToolMessage
    #   Loops until LLM stops calling tools (no tool_calls)
    # ─────────────────────────────────────────────────────

    def _build_react_graph(self):
        tools = self._build_tools()
        return create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=SYSTEM_PROMPT,   # injected as SystemMessage[0]
        )

    # ─────────────────────────────────────────────────────
    # STREAMING RUN
    # ─────────────────────────────────────────────────────

    async def stream_run(self, user_message: str) -> AsyncIterator[dict]:
        """
        Run the ReAct TAO loop for one user turn and stream events.

        The full chat_history is passed in every invocation so the LLM
        has complete context when reasoning about which stage the user is at.

        Yields
        ──────
        {"type": "status",   "content": str}           — tool being called
        {"type": "products", "ids": [...], ...}         — after search_products
        {"type": "message",  "content": str, ...}       — final AI reply
        {"type": "error",    "content": str}            — on failure
        """
        react_graph = self._build_react_graph()

        # Build the message list: history + this turn's user message
        input_messages: list[BaseMessage] = [
            *self.chat_history,
            HumanMessage(content=user_message),
        ]

        STATUS = {
            "ask_stylist":    "🎨 Consulting your personal stylist…",
            "search_products":"🔍 Searching our collection…",
            "handle_cart":    "🛒 Updating your cart…",
            "ask_rag":        "📚 Checking our knowledge base…",
        }

        final_output:     str | None  = None
        emitted_products: bool        = False
        last_ai_text:     str | None  = None   # fallback if chain end misses output

        try:
            async for chunk in react_graph.astream(
                {"messages": input_messages},
                stream_mode="updates",
            ):
                # chunk = {node_name: {"messages": [...]}}
                if not isinstance(chunk, dict):
                    continue

                for node_name, node_update in chunk.items():
                    if not isinstance(node_update, dict):
                        continue

                    messages: list = node_update.get("messages", [])

                    for msg in messages:

                        # ── Agent node: LLM decided to call a tool ──────
                        if isinstance(msg, AIMessage):
                            tool_calls = getattr(msg, "tool_calls", None) or []
                            if tool_calls:
                                # Emit a status bubble for each tool about to be called
                                for tc in tool_calls:
                                    status = STATUS.get(tc["name"], "⚙️ Working…")
                                    yield {"type": "status", "content": status}
                            else:
                                # LLM produced a final text response — capture it
                                text = _extract_ai_text(msg)
                                if text:
                                    last_ai_text = text

                        # ── Tools node: result of a tool call ───────────
                        else:
                            raw = _msg_text(msg)
                            try:
                                parsed = json.loads(raw)
                            except (json.JSONDecodeError, TypeError):
                                continue

                            if not isinstance(parsed, dict):
                                continue

                            # Emit product cards immediately when search returns
                            if "product_ids" in parsed and not emitted_products:
                                emitted_products = True
                                yield {
                                    "type":       "products",
                                    "ids":        parsed.get("product_ids", []),
                                    "image_urls": parsed.get("image_urls", []),
                                    "message":    f"Found {parsed.get('count', 0)} styles.",
                                }

        except Exception as exc:
            print(f"[Orchestrator] Error during stream_run: {exc}")
            yield {"type": "error", "content": f"Something went wrong: {exc}"}
            return

        # ── Resolve final output ───────────────────────────────────────
        # LangGraph's astream(stream_mode="updates") surfaces the final
        # AIMessage inside the "agent" node update — captured as last_ai_text.
        final_output = last_ai_text

        if final_output:
            # Persist this turn in chat history for future turns
            self.chat_history.append(HumanMessage(content=user_message))
            self.chat_history.append(AIMessage(content=final_output))

            yield {
                "type":            "message",
                "content":         final_output,
                "recommended_ids": self.recommended_ids,
            }
        else:
            yield {
                "type":    "error",
                "content": "I couldn't generate a response. Please try again.",
            }