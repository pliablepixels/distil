"""LangGraph + Distil — in-process, no proxy.

LangGraph agents accumulate messages in graph state and re-send the whole list
every step — exactly the re-send-the-world cost Distil targets. Distil ships a
drop-in ``pre_model_hook`` (LangGraph's seam for transforming state right before
the LLM node) that compresses the message list reversibly, in-process, with no
network hop.

    pip install langgraph langchain-anthropic

The hook is duck-typed: Distil never imports langgraph/langchain, so this works
across versions. It returns ``{"messages": <compressed>}`` — only the message
list is updated; every other state field is left untouched.
"""

import os

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from distil.integrations.langgraph import pre_model_hook

model = ChatAnthropic(
    model="claude-opus-4-5",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)


def get_weather(city: str) -> str:
    """A toy tool whose verbose output is what bloats the trajectory."""
    return (
        f"It is 21C and sunny in {city}. "
        + "Detailed hourly breakdown: "
        + "; ".join(f"{h:02d}:00 21C clear" for h in range(24))
    )


# Compress state messages right before every model call. verbatim=True keeps
# byte-exact Tier-0 only (no digests); omit it for the default reversible tier.
agent = create_react_agent(
    model,
    tools=[get_weather],
    pre_model_hook=pre_model_hook(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Paris and Tokyo?"}]}
)
print(result["messages"][-1].content)

# Manual alternative — call compress_state inside any node yourself:
#
#   from distil.integrations.langgraph import compress_state
#
#   def my_model_node(state):
#       state = compress_state(state, verbatim=True)
#       ...
