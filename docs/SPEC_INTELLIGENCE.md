# SPEC: Advanced Insight Engines

本ドキュメントは、Chat Log Analyzer における「深いインサイト」を導き出すための 4 本柱（Core Pillars）の技術詳細と実装設計を定義します。

---

## 1. Delta分析 (DeltaAnalysisEngine)
**目的**: 異なるコホート（例：継続者 vs 離脱者）間の単語・トピック出現傾向の「差分」を特定する。

### 技術アプローチ
- **TF-IDF Log Odds Ratio**: 特定のグループで有意に高い出現率を持つ単語を抽出。
- **Topic Distribution Shift**: クラスタ分布のグループ間比較（Kullback-Leibler 散漫度等）。
- **Implementation**: `src/analysis/delta_analysis.py`

### アウトプット
- `Characteristic Keywords`: 特定グループを象徴する単語リスト。
- `Cohort Comparison Chart`: 属性ごとの話題の偏りを示すヒートマップ。

---

## 2. アーキタイプ分類 (ArchetypeEngine)
**目的**: 会話のスタイル、量、感情、語彙からユーザーを類型化し、ペルソナを定義する。

### 技術アプローチ
- **Feature Engineering**:
    - `Engagement`: 発話頻度、平均文字数、返信間隔。
    - `Sentiment`: ポジティブ/ネガティブ比率の平均。
    - `Vocabulary`: 語彙の多様性 (Type-Token Ratio)。
- **Clustering**: 上記特徴量に対する K-means または手法の適用。
- **Implementation**: `src/analysis/archetype.py`

### アウトプット
- `User Archetype Map`: ユーザー性格・傾向の散布図。
- `Persona Descriptions`: LLM による各タイプの定性的な特徴説明。

---

## 3. Gap分析 (GapDiscoveryEngine)
**目的**: ユーザーが明示していない「潜在的な不満・要望」や「情報の欠落」を LLM で推論する。

### 技術アプローチ
- **Hybrid LLM Backend**:
    - **Local Inference (Default)**: [BitNet b1.58 2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) を採用。1.58-bit 量子化により低リソース・高速・低コストなローカル分析を実現。
    - **API Inference (Optional)**: クライアントの要望に応じて OpenAI API (GPT-5等) へ切り替え可能なプラグイン構成。
- **Contextual Reasoning**: 
    - LLM に対し「ユーザーが繰り返し触れているが、解決策が示されていないトピック」を抽出させる。
    - 「期待 (Expectation) vs 現状 (Reality)」の不一致を特定する。
    - ※現在はルールベースのヒューリスティック分析を実装済み。LLM連携は拡張予定。
- **Implementation**: `src/analysis/gap_analysis.py`

### アウトプット
- `Opportunity Score`: 改善によって得られるインパクトの推定。
- `Implicit Needs List`: 潜在ニーズの言語化。

---

## 4. モーメント分析 (MomentAnalysisEngine)
**目的**: セッション内での感情や熱量の決定的な変化点（転換点）を特定する。

### 技術アプローチ
- **Time-series Sentiment Analysis**: 1セッション内での移動平均感情スコアの算出。
- **Vibe Shift Detection**: 感情やトピックが急激に変化した箇所の勾配検知。
- **Qualitative Evaluation**: "Aha! Moment"（納得）、"Friction"（摩擦）、"Churn Signal"（離脱）の分類。
- **Implementation**: `src/analysis/moment_analysis.py`

### アウトプット
- `Session Vibe Graph`: 1会話内の感情・熱量の折れ線グラフ。
- `Critical Moments`: インサイトの要となる発話のタイムスタンプ特定。

---

## 共通指標: エンゲージメントの質的評価 (Dialogue Depth Score)
全エンジンで使用される基本指標。
- **計算式イメージ**: `(Topic Continuity * Vocabulary Density) / (Response Jitters)`
- **継続性**: 1つの話題がどれだけ深掘りされたか。
- **密度**: 情報量（文字数だけでなく重要語の割合）。
- **ノイズ**: 意味のない短文の発話比率。

---

## 5. インタラクティブ・プレゼンテーション層 (UX)

分析結果を「静止画」ではなく、クライアントが自身で探索できる「対話型資料」として提供します。

### A. Jupyter Interactive Showcase (`notebooks/*.ipynb`)
**目的**: エンジニアや専門家がパラメータを調整しながら深掘りを行う「生きた報告書」。
- **IPyWidgets 活用**:
    - **Data Range Selector**: カレンダー UI による分析期間のフィルタリング。
    - **Target Segment Dropdown**: 全体、年代別、離脱グループ等の切り替え。
    - **Parameter Sliders**: クラスタ数や LLM 要約の粒度をリアルタイム調整。
- **Auto-Reload**: ソースコードの変更を即座に反映。

### B. Client-Safe Dashboard (`src/app/main.py`)
**目的**: クライアント提供やプロトタイプ展示用の、クリーンな Web ダッシュボード。
- **Streamlit フレームワーク**:
    - **Zero-Code Operation**: 技術に詳しくないユーザーでもサイドバー操作だけで全分析を確認。
    - **Downloadable Reports**: 分析結果の CSV・画像エクスポート。
    - **Context Masking**: クライアント向けに不要な技術ログを表示せず、インサイトのみを洗練された UI で提供。

---

## 6. デプロイメント・環境戦略 (SageMaker Optimized)

Amazon SageMaker 環境での動作を第一優先とし、ローカル開発と本番環境の乖離を最小限にします。

### A. 環境構築の方針
- **No-Docker Principle**: SageMaker Lifecycle Config または Conda 仮想環境によるセットアップ。SageMaker ノートブック（Linux）のネイティブ性能を最大限活用。
- **Library Stability**:
    - Windows 環境で不安定な UMAP/HDBSCAN は、SageMaker 環境（Linux）では安定動作。
    - ローカル（Windows）開発時は、自動的に PCA/K-Means 等の軽量フォールバックへ切り替わる「環境適応型」ロジックを採用。

### B. ベクトルモデルの柔軟性
- **Default Model**: 日本語性能に定評のある `intfloat/multilingual-e5-base` または `sonoisa/sentence-bert-base-ja-mean-tokens` を採用。
- **Customizable**: `config.py` の設定変更のみで最新のモデル（HuggingFace 上の BERT/RoBERTa 等）に容易に差し替え可能。
