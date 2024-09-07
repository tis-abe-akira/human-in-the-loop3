import os
import sys
from dotenv import load_dotenv

from json import load
from typing import TypedDict, Literal, override
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
# from IPython.display import Image, display


@tool
def weather_search(city: str):
    """Search for the weather"""
    print("----")
    print(f"Searching for: {city}")
    print("----")
    return "Sunny!"


class State(MessagesState):
    """Simple state."""


def call_llm(state):
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools([weather_search])
    return {"messages": [model.invoke(state["messages"])]}


def human_review_node(state):
    pass


def run_tool(state):
    new_messages = []
    tools = {"weather_search": weather_search}
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        new_messages.append(
            {
                "role": "tool",
                "name": tool_call["name"],
                "content": result,
                "tool_call_id": tool_call["id"],
            }
        )
    return {"messages": new_messages}


def route_after_llm(state) -> Literal[END, "human_review_node"]:
    if len(state["messages"][-1].tool_calls) == 0:
        return END
    else:
        return "human_review_node"


def route_after_human(state) -> Literal["run_tool", "call_llm"]:
    if isinstance(state["messages"][-1], AIMessage):
        return "run_tool"
    else:
        return "call_llm"

def create_graph():
    builder = StateGraph(State)
    builder.add_node(call_llm)
    builder.add_node(run_tool)
    builder.add_node(human_review_node)
    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", route_after_llm)
    builder.add_conditional_edges("human_review_node", route_after_human)
    builder.add_edge("run_tool", "call_llm")

    # Set up memory
    memory = MemorySaver()

    # Add
    return builder.compile(checkpointer=memory, interrupt_before=["human_review_node"])

    # View
    # display(Image(graph.get_graph().draw_mermaid_png()))

def main():
    load_dotenv(override=True)

    graph = create_graph()

    # Input
    # initial_input = {"messages": [{"role": "user", "content": "hi!"}]}
    initial_input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}


    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print(event)

    print("Pending Executions!")
    print(graph.get_state(thread).next)

    user_input = input("Human Review: ")
    if user_input == "yes":
        print("approved")
        for event in graph.stream(None, thread, stream_mode="values"):
            print(event)
    else:
        print("rejected")


if __name__ == "__main__":
    main()
