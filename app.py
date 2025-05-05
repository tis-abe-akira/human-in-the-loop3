from typing import Any
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import HumanInTheLoopAgent


def show_messages(messages: list[Any]) -> None:
    """メッセージリストをStreamlit上に表示する

    Args:
        messages (list[Any]): 表示するメッセージのリスト。HumanMessage、AIMessage、ToolMessageのいずれかのインスタンス。

    Raises:
        ValueError: 未知のメッセージタイプが含まれている場合に発生
    """
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message(message.type):
                st.write(message.content)

        elif isinstance(message, AIMessage):
            # tool_callの場合はツールの承認を求める旨を表示
            if len(message.tool_calls) != 0:
                for tool_call in message.tool_calls:
                    with st.chat_message(message.type):
                        st.write("エージェントがツールの承認を求めています")
                        st.write(f"ツール名: {tool_call['name']}")
                        st.write(f"引数: {tool_call['args']}")
            else:
                with st.chat_message(message.type):
                    st.write(message.content)

        elif isinstance(message, ToolMessage):
            with st.chat_message(message.type):
                st.write("ツールの実行結果")
                st.write(message.content)

        else:
            raise ValueError(f"Unknown message type: {type(message)}")


def app() -> None:
    """Streamlitアプリケーションのメインエントリーポイント

    環境変数を読み込み、Human-in-the-loopエージェントを初期化し、
    ユーザーとエージェントの対話インターフェースを提供する。
    
    機能:
    - エージェントの状態管理
    - グラフの可視化
    - メッセージ履歴の表示
    - ユーザー入力の処理
    - ツール実行の承認機能
    """
    load_dotenv(override=True)

    st.title("LangGraphでのHuman-in-the-loopの実装")

    # st.session_stateにagentを保存
    if "agent" not in st.session_state:
        st.session_state.agent = HumanInTheLoopAgent()
    agent = st.session_state.agent

    # グラフを表示
    with st.sidebar:
        st.image(agent.mermaid_png())

    # st.session_stateにthread_idを保存
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex
    thread_id = st.session_state.thread_id
    st.write(f"thread_id: {thread_id}")

    # ユーザーの指示を受け付ける
    human_message = st.chat_input()
    if human_message:
        with st.spinner():
            agent.handle_human_message(human_message, thread_id)

    # 会話履歴を表示
    messages = agent.get_messages(thread_id)
    show_messages(messages)

    # 次がhuman_review_nodeの場合は承認ボタンを表示
    if agent.is_next_human_review_node(thread_id):
        approved = st.button("承認")
        # 承認されたらエージェントを実行
        if approved:
            with st.spinner():
                agent.handle_approve(thread_id)
            # 会話履歴を表示するためrerun
            st.rerun()


app()
