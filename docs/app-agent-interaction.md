# app.py と agent.py のインタラクション

このドキュメントは、StreamlitアプリケーションとHuman-in-the-loopエージェント間の詳細なインタラクションフローを示します。

## 全体的なインタラクションシーケンス

```mermaid
sequenceDiagram
    participant User
    participant Streamlit as app.py (Streamlit)
    participant Agent as agent.py (HumanInTheLoopAgent)
    participant LLM as ChatOpenAI
    participant Tool as amortization_calculation

    Note over Streamlit: アプリケーション初期化
    Streamlit->>Agent: HumanInTheLoopAgent()を初期化
    Agent-->>Streamlit: エージェントインスタンス返却
    
    Note over Streamlit: サイドバーにグラフ表示
    Streamlit->>Agent: mermaid_png()
    Agent-->>Streamlit: グラフのPNG画像
    
    Note over User,Tool: 通常の質問応答フロー
    User->>Streamlit: チャット入力（非ツール質問）
    Streamlit->>Agent: handle_human_message(human_message, thread_id)
    Agent->>LLM: _call_llm() - メッセージ送信
    LLM-->>Agent: 通常の応答（ツール呼び出しなし）
    Agent-->>Streamlit: 処理完了
    
    Streamlit->>Agent: get_messages(thread_id)
    Agent-->>Streamlit: メッセージ履歴取得
    Streamlit->>Agent: is_next_human_review_node(thread_id)
    Agent-->>Streamlit: False（承認不要）
    Streamlit-->>User: 応答表示

    Note over User,Tool: ツール使用時の承認フロー
    User->>Streamlit: チャット入力（ローン計算質問）
    Streamlit->>Agent: handle_human_message(human_message, thread_id)
    Agent->>LLM: _call_llm() - メッセージ送信
    LLM-->>Agent: ツール呼び出し付き応答
    Note over Agent: _route_after_llm()でhuman_review_nodeに遷移
    Agent-->>Streamlit: 処理完了（human_review_nodeで中断）
    
    Streamlit->>Agent: get_messages(thread_id)
    Agent-->>Streamlit: メッセージ履歴取得
    Streamlit->>Agent: is_next_human_review_node(thread_id)
    Agent-->>Streamlit: True（承認待ち状態）
    Streamlit-->>User: ツール承認ボタン表示

    User->>Streamlit: 承認ボタンクリック
    Streamlit->>Agent: handle_approve(thread_id)
    Agent->>Tool: _run_tool() - ツール実行
    Tool-->>Agent: 計算結果返却
    Agent->>LLM: _call_llm() - 結果を含むメッセージ送信
    LLM-->>Agent: 結果の説明応答
    Agent-->>Streamlit: 処理完了
    
    Note over Streamlit: st.rerun()でUI更新
    Streamlit->>Agent: get_messages(thread_id)
    Agent-->>Streamlit: 更新されたメッセージ履歴
    Streamlit->>Agent: is_next_human_review_node(thread_id)
    Agent-->>Streamlit: False（承認完了）
    Streamlit-->>User: 最終結果表示

    Note over User,Tool: ツール実行拒否フロー
    User->>Streamlit: チャット入力（ローン計算質問）
    Streamlit->>Agent: handle_human_message(human_message, thread_id)
    Agent->>LLM: _call_llm() - メッセージ送信
    LLM-->>Agent: ツール呼び出し付き応答
    Agent-->>Streamlit: 処理完了（human_review_nodeで中断）
    
    Streamlit->>Agent: is_next_human_review_node(thread_id)
    Agent-->>Streamlit: True（承認待ち状態）
    Streamlit-->>User: ツール承認ボタン表示

    User->>Streamlit: 新しいメッセージ入力（拒否の意図）
    Note over Streamlit: is_next_human_review_node()がTrueのため、拒否として処理
    Streamlit->>Agent: handle_human_message(reject_message, thread_id)
    Note over Agent: ToolMessage(status="error")を生成
    Agent->>Agent: update_state() - エラーメッセージを状態に追加
    Agent->>LLM: _call_llm() - 拒否された旨を送信
    LLM-->>Agent: 代替提案など
    Agent-->>Streamlit: 処理完了
    
    Streamlit->>Agent: get_messages(thread_id)
    Agent-->>Streamlit: 更新されたメッセージ履歴
    Streamlit-->>User: 代替提案表示
```

## 重要なメソッドの詳細

### app.py側のメソッド

1. **show_messages(messages)**
   - HumanMessage、AIMessage、ToolMessageの種類に応じて表示を分岐
   - AIMessageにtool_callsがある場合、承認求む旨を表示

2. **app()メインループ**
   - エージェントの初期化とsession_stateでの管理
   - グラフの可視化（サイドバー）
   - メッセージ処理とUI更新

### agent.py側のメソッド

1. **handle_human_message(human_message, thread_id)**
   - 承認待ち状態でのメッセージはツール拒否として処理
   - graph.stream()でワークフローを実行

2. **handle_approve(thread_id)**
   - Command(resume="approve")でワークフロー再開

3. **is_next_human_review_node(thread_id)**
   - 状態の`next`フィールドをチェック
   - UI側の承認ボタン表示制御に使用

4. **get_messages(thread_id)**
   - 現在の会話履歴を取得
   - UI更新のたびに呼び出される

## 状態管理の流れ

- **thread_id**: セッション単位で会話を識別
- **StateSnapshot**: LangGraphの状態管理でcheckpointを保存
- **interrupt_before**: human_review_nodeで自動的に中断
- **MemorySaver**: 会話履歴を永続化