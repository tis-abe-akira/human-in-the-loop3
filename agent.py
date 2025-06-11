from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.pregel.types import StateSnapshot
from langgraph.types import Command


@tool
def amortization_calculation(principal: int, annual_interest_rate: float, num_payments: int) -> int:
    """住宅ローンの毎月の返済額を計算するツール

    Args:
        principal (int): 借入金額（元金）
        annual_interest_rate (float): 年間利率（%）
        num_payments (int): 返済回数（月数）

    Returns:
        int: 毎月の返済額（円）、小数点以下切り捨て
    """
    # 月利率の計算
    monthly_interest_rate = annual_interest_rate / 1200
    # 毎月の返済額を計算する式
    monthly_payment = principal * (monthly_interest_rate * (1 + monthly_interest_rate) ** num_payments) / ((1 + monthly_interest_rate) ** num_payments - 1)
    # 計算結果を切り捨てて整数に変換
    return int(monthly_payment)


class HumanInTheLoopAgentState(MessagesState):
    """エージェントの状態を管理するクラス

    MessagesStateを継承し、メッセージの履歴を保持する。
    """


class HumanInTheLoopAgent:
    """Human-in-the-loopエージェントの実装

    このエージェントは以下の機能を提供する：
    - LLMによる応答生成
    - ツールの実行
    - 人間によるレビューと承認
    - 状態管理とチェックポイント
    """
    def __init__(self) -> None:
        builder = StateGraph(HumanInTheLoopAgentState)
        builder.add_node("call_llm", self._call_llm)
        builder.add_node("run_tool", self._run_tool)
        builder.add_node("human_review_node", self._human_review_node)
        builder.add_edge(START, "call_llm")
        builder.add_conditional_edges("call_llm", self._route_after_llm)
        builder.add_conditional_edges("human_review_node", self._route_after_human)
        builder.add_edge("run_tool", "call_llm")

        memory = MemorySaver()

        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["human_review_node"],
        )

    def _call_llm(self, state: dict) -> dict:
        """LLMを呼び出してメッセージを生成する

        Args:
            state (dict): 現在の状態

        Returns:
            dict: 生成されたメッセージを含む新しい状態
        """
        model = ChatOpenAI(model="gpt-4o-mini").bind_tools([amortization_calculation])
        return {"messages": [model.invoke(state["messages"])]}

    def _human_review_node(self, state: dict) -> None:
        """人間によるレビューノード

        ツールの実行前に人間の承認を待機する。

        Args:
            state (dict): 現在の状態
        """
        pass

    def _run_tool(self, state: dict) -> dict:
        """承認されたツールを実行する

        Args:
            state (dict): 現在の状態（実行するツールの情報を含む）

        Returns:
            dict: ツールの実行結果を含む新しい状態
        """
        new_messages = []
        tools = {"amortization_calculation": amortization_calculation}
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

    def _route_after_llm(self, state: dict) -> Literal[END, "human_review_node"]:
        """LLM実行後の遷移先を決定する

        Args:
            state (dict): 現在の状態

        Returns:
            Literal[END, "human_review_node"]: 
                - ツールの呼び出しがある場合はhuman_review_node
                - ない場合はEND
        """
        if len(state["messages"][-1].tool_calls) == 0:
            return END
        else:
            return "human_review_node"

    def _route_after_human(self, state: dict) -> Literal["run_tool", "call_llm"]:
        """人間のレビュー後の遷移先を決定する

        Args:
            state (dict): 現在の状態

        Returns:
            Literal["run_tool", "call_llm"]: 
                - AIMessageの場合はrun_tool
                - それ以外の場合はcall_llm
        """
        if isinstance(state["messages"][-1], AIMessage):
            return "run_tool"
        else:
            return "call_llm"

    def handle_human_message(self, human_message: str, thread_id: str) -> None:
        """ユーザーからのメッセージを処理する

        Args:
            human_message (str): ユーザーからのメッセージ
            thread_id (str): 会話を識別するためのID
        """
        # 承認待ちの状態でhuman_messageが送信されるのは、ツールの呼び出しを修正したい状況
        # そのため、次がhuman_review_nodeの場合、ツールの呼び出しが失敗したことをStateに追加
        # 参考: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/review-tool-calls/#give-feedback-to-a-tool-call
        if self.is_next_human_review_node(thread_id):
            last_message = self.get_messages(thread_id)[-1]
            tool_reject_message = ToolMessage(
                content="Tool call rejected",
                status="error",
                name=last_message.tool_calls[0]["name"],
                tool_call_id=last_message.tool_calls[0]["id"],
            )
            self.graph.update_state(
                config=self._config(thread_id),
                values={"messages": [tool_reject_message]},
                as_node="human_review_node",
            )

        for _ in self.graph.stream(
            input={"messages": [HumanMessage(content=human_message)]},
            config=self._config(thread_id),
            stream_mode="values",
        ):
            pass

    def handle_approve(self, thread_id: str) -> None:
        """ツールの実行を承認する

        Args:
            thread_id (str): 会話を識別するためのID
        """
        for _ in self.graph.stream(
            Command(resume="approve"),
            config=self._config(thread_id),
            stream_mode="values",
        ):
            pass

    def get_messages(self, thread_id):
        """会話履歴を取得する

        Args:
            thread_id: 会話を識別するためのID

        Returns:
            list: メッセージのリスト。存在しない場合は空リストを返す
        """
        state = self._get_state(thread_id)
        return state.values.get("messages", [])  # "messages"キーが見つからない場合は空のリストを返す

    def is_next_human_review_node(self, thread_id: str) -> bool:
        """次のノードが人間によるレビューノードかどうかを判定する

        Args:
            thread_id (str): 会話を識別するためのID

        Returns:
            bool: 次のノードがhuman_review_nodeの場合はTrue
        """
        graph_next = self._get_state(thread_id).next
        return len(graph_next) != 0 and graph_next[0] == "human_review_node"

    def _get_state(self, thread_id: str) -> StateSnapshot:
        return self.graph.get_state(config=self._config(thread_id))

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def mermaid_png(self) -> bytes:
        """グラフをMermaid形式のPNG画像として取得する

        Returns:
            bytes: PNG画像のバイトデータ
        """
        return self.graph.get_graph().draw_mermaid_png()
