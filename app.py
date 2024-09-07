from pdb import run
import uu
import streamlit as st
from uuid import uuid4

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

def run_agent(graph, graph_input, thread):
    for event in graph.stream(graph_input, thread, stream_mode="values"):
        last_message = event["messages"][-1]
        if isinstance(last_message, AIMessage):
            st.write("AI Response!")
            st.write(last_message)
        # st.write(event)


def app():
    load_dotenv(override=True)
    st.title("LangGraphでHuman-in-the-loopを実現する")

    # st.session_stateにthread_idを保存
    if "thread_id" not in st.session_state:
        thread_id = uuid4().hex
        st.session_state.thread_id = thread_id
    thread_id = st.session_state.thread_id

    graph = create_graph()

    user_message = st.text_input("")

    if not user_message:
        # return
        st.stop()

    st.write(f"User Input: {user_message}")


    # Input
    # initial_input = {"messages": [{"role": "user", "content": "hi!"}]}
    initial_input = {"messages": [{"role": "user", "content": user_message}]}

    # Thread
    thread = {"configurable": {"thread_id": thread_id}}

    # Run the graph until the first interruption
    run_agent(graph, initial_input, thread)

    # st.write("Pending Executions!")
    next_node = graph.get_state(thread).next[0]
    st.write(graph.get_state(thread).next)
    if next_node != "human_review_node":
        st.stop()

    # 承認ボタンを設置
    approved = st.button("Approve")

    if not approved:
        st.stop()

    run_agent(graph, None, thread)


app()
