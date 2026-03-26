"""
Stylist Agent — Conversational fashion consultant.

Maintains its own message history across turns.
Returns structured JSON: {"type": "question"/"suggestion", ...}
"""

import json
import re
from typing import Annotated, List

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict

from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

COSTUME_STYLIST_PROMPT = """
You are a professional personal costume consultant and stylist specializing in fashion for women aged 18–35.

Your goal is to deeply understand the user and recommend the most suitable dress tailored specifically to their situation.

-------------------------
DECISION LOGIC (CRITICAL)
-------------------------
1. Extract the following attributes from the user's message:
   - occasion
   - time_of_day
   - style_preference
   - fit_preference

2. Count how many of these attributes are clearly known.

3. IF fewer than 3 attributes are known:
   → Ask 1–2 short, highly relevant clarifying questions.
   → DO NOT give any recommendation.

4. IF 3 or more attributes are known:
   → Generate a dress recommendation.

-------------------------
PERSONALIZATION RULES
-------------------------
- Act like a real stylist having a natural conversation.
- Keep questions short, specific, and useful.
- Understand emotional intent:
  - Is this a special occasion?
  - Do they want to stand out or stay subtle?

-------------------------
RECOMMENDATION RULES
-------------------------
- Focus ONLY on the dress (no accessories, footwear, or styling extras).
- Ensure the dress is realistic, trendy, and appropriate.

CLIP OPTIMIZATION (IMPORTANT):
- The description must be visually rich and embedding-friendly.
- Include:
  - color
  - silhouette/fit (A-line, bodycon, etc.)
  - length (mini, midi, maxi)
  - fabric/material (satin, cotton, chiffon, etc.)
  - dress type/details (floral, ruffled, slit, etc.)

- Use strong fashion keywords.
- Avoid vague words (nice, pretty, stylish).
- Avoid metaphors or poetic language.

-------------------------
OUTPUT FORMAT (STRICT)
-------------------------

ALWAYS return a valid JSON object with NO markdown wrapping.

If asking questions:
{
  "type": "question",
  "questions": "<comforting message & question for the user>"
}

If recommending:
{
  "type": "suggestion",
  "suggestion": "<ONE sentence describing ONLY the dress>",
  "message": "<ONE short reassuring sentence>",
  "follow_up": "<ONE short sentence asking which they like and encouraging interaction>",
  "user_preference": "<A concise summary of the user preferences>"
}
""".strip()


# ─────────────────────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────────────────────

class StylistState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# ─────────────────────────────────────────────────────────────
# AGENT CLASS
# ─────────────────────────────────────────────────────────────

class StylistAgent:
    """
    Conversational stylist that maintains its own history.
    Call `chat(user_message, history)` on every turn.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7)
        self._graph = self._build_graph()

    # ── Graph ──────────────────────────────────────────────

    def _build_graph(self):
        def stylist_node(state: StylistState):
            messages = state["messages"]
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=COSTUME_STYLIST_PROMPT)] + messages
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(StylistState)
        graph.add_node("stylist", stylist_node)
        graph.set_entry_point("stylist")
        graph.add_edge("stylist", END)
        return graph.compile()

    # ── Public API ─────────────────────────────────────────

    async def chat(
        self,
        user_message: str,
        history: list[dict],
    ) -> dict:
        """
        Parameters
        ----------
        user_message : str
            The latest user input.
        history : list[dict]
            Serialized history: [{"role": "human"/"ai", "content": "..."}, ...]

        Returns
        -------
        dict
            {
              "response": <parsed JSON from stylist>,
              "history":  <updated serialized history>,
              "raw":      <raw LLM output string>
            }
        """
        log.info("[Stylist] chat called — history_turns=%d  message=%r",
                 len(history), user_message)

        # Reconstruct LangChain messages from serialized history
        lc_messages: list[AnyMessage] = []
        for msg in history:
            if msg["role"] == "human":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                lc_messages.append(AIMessage(content=msg["content"]))

        lc_messages.append(HumanMessage(content=user_message))

        # Invoke graph
        result = await self._graph.ainvoke({"messages": lc_messages})
        raw_response: str = result["messages"][-1].content
        log.debug("[Stylist] raw LLM response: %s", raw_response[:200])

        # Parse JSON (handle markdown code fences)
        parsed = self._parse_json(raw_response)
        log.info("[Stylist] parsed response type=%s", parsed.get("type", "?"))

        # Update serialized history
        new_history = history + [
            {"role": "human", "content": user_message},
            {"role": "ai",    "content": raw_response},
        ]

        return {"response": parsed, "history": new_history, "raw": raw_response}

    # ── Helpers ────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Strip markdown fences and parse JSON."""
        text = text.strip()
        # Remove ```json ... ``` or ``` ... ```
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            log.warning("[Stylist] JSON parse failed, falling back to plain text question")
            # Fallback: treat as plain text question
            return {"type": "question", "questions": text}
