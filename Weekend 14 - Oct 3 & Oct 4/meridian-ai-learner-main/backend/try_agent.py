"""One tool, one agent: print the message trail (run from backend/):  uv run python try_agent.py"""
from langchain.agents import create_agent
from langchain_core.tools import tool

from rag.llm import extract_text, get_llm
from agent.tools import validate_fx_hedge


@tool
def approval_needed(amount_eur: float) -> str:
    """Says who must approve a purchase of this amount, in euros."""
    if amount_eur > 1_000_000:
        return "CFO and Management Board"
    if amount_eur > 250_000:
        return "Department Head"
    return "The requesting manager"


agent = create_agent(
    model=get_llm(),
    tools=[approval_needed,validate_fx_hedge],
    system_prompt="You are a procurement assistant. Always use your tool. Do not guess.",
    
)

result = agent.invoke({"messages": [{"role": "user", "content": "Can you tell me how much is 1 eur to inr?, you have the validate_fx_hedge tool, please use it to check the rate and tell me if I need approval for 500000 eur purchase."}]})

for message in result["messages"]:
    calls = getattr(message, "tool_calls", None)
    print(f"{message.type}", calls if calls else extract_text(message.content))
