# human-in-the-loop3

LangGraphのHuman in the Loopの実装です。
StudyCoの勉強会の内容を写経したうえで、ツール部分の実装をファイナンス系の計算に変更しています。


##　起動方法

```bash
uv run streamlit run app.py
```

## 画面操作イメージ

```bash
年利2.4%で30万円を36回払いするとどうなる？
```

![screen-image](./pics/screen-image.png)

## インタラクションの流れ

### 通常の質問応答の場合

```mermaid
sequenceDiagram
    actor User
    participant Streamlit as Streamlit UI
    participant Agent
    participant LLM

    User->>Streamlit: メッセージを入力
    Streamlit->>Agent: handle_human_message()
    Agent->>LLM: 質問内容を送信
    LLM-->>Agent: 応答を返す
    Agent-->>Streamlit: メッセージを更新
    
    Streamlit->>Agent: get_messages()
    Agent-->>Streamlit: メッセージ履歴
    Streamlit->>Agent: is_next_human_review_node()
    Agent-->>Streamlit: false
    Streamlit-->>User: 応答を表示
```

### ツール使用時の承認フロー

```mermaid
sequenceDiagram
    actor User
    participant Streamlit as Streamlit UI
    participant Agent
    participant LLM
    participant Tool as 計算ツール

    User->>Streamlit: ローン計算の質問
    Streamlit->>Agent: handle_human_message()
    Agent->>LLM: 質問内容を送信
    LLM-->>Agent: ツール使用を提案
    Agent-->>Streamlit: ツール承認を要求
    
    Streamlit->>Agent: get_messages()
    Agent-->>Streamlit: メッセージ履歴
    Streamlit->>Agent: is_next_human_review_node()
    Agent-->>Streamlit: true
    Streamlit-->>User: 承認ボタンを表示

    User->>Streamlit: 承認ボタンをクリック
    Streamlit->>Agent: handle_approve()
    Agent->>Tool: ツールを実行
    Tool-->>Agent: 計算結果を返す
    Agent->>LLM: 結果を送信
    LLM-->>Agent: 結果を説明
    
    Streamlit->>Agent: get_messages()
    Agent-->>Streamlit: 更新されたメッセージ履歴
    Streamlit->>Agent: is_next_human_review_node()
    Agent-->>Streamlit: false
    Streamlit-->>User: 最終結果を表示
```

これらのシーケンス図は、アプリケーションの主要なインタラクションフローを示しています：

1. get_messages(): 
   - 現在のスレッドの会話履歴を取得
   - UI更新のたびに呼び出され、最新の会話状態を表示

2. is_next_human_review_node():
   - 次のステップが人間の承認を必要とするかを判断
   - 承認ボタンの表示/非表示の制御に使用

これらの内部メソッドは、ユーザーインターフェースの状態管理と表示制御に重要な役割を果たしています。
