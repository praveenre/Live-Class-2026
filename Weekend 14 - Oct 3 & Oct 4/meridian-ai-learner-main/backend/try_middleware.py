"""See two guard-rails work (run from backend/):  uv run python try_middleware.py"""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, PIIMiddleware
from langchain_core.tools import tool

from rag.llm import extract_text, get_llm


@tool
def approval_needed(amount_eur: float) -> str:
    """Says who must approve a purchase of this amount, in euros."""
    return "Department Head" if amount_eur > 250_000 else "The requesting manager"


def last_text(agent, question):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return extract_text(result["messages"][-1].content)


# 1. Privacy: the model never sees the e-mail address
request = (
    "Message: please contact anna.schmidt@supplier.example about the order. "
    "Task: reply with only the e-mail address that appears in the message."
)
redact = PIIMiddleware("email", strategy="redact", apply_to_input=True)
plain = create_agent(model=get_llm(), tools=[])
guarded = create_agent(model=get_llm(), tools=[], middleware=[redact])
print("without the guard:", last_text(plain, request))
print("with the guard   :", last_text(guarded, request))

# 2. Cost: one model call is allowed, so the agent ends politely instead of looping
limit = ModelCallLimitMiddleware(run_limit=1, exit_behavior="end")
limited = create_agent(model=get_llm(), tools=[approval_needed], middleware=[limit])
print("\nwith a limit of one model call:", last_text(limited, "Who must approve 480,000 EUR? Use your tool."))
