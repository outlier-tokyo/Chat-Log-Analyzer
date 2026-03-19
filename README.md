# Chat Log Analyzer

会話ログデータを多角的に分析・可視化し、深いインサイトを導き出すための高度な Python フレームワークです。
複数のデータソースに対応し、テキストマイニング、自然言語処理（NLP）、そして LLM を組み合わせた一連の分析パイプラインを提供します。

## 🚀 Overview

本プロジェクトは、さまざまなチャット形式のログ（CSV、データベース、HuggingFace Datasets など）を統合的に扱い、対話の構造や傾向を解き明かすことを目的としています。

**主な特徴:**
*   **データソース非依存:** ローカルの CSV から AWS Athena (S3)、公開データセットまで柔軟に対応。
*   **多層的な分析パイプライン:** 
    *   **形態素解析 & クリーニング:** 日本語テキストの最適な正規化。
    *   **ベクトル解析:** Sentence-BERT を用いた意味レベルでの埋め込み。
    *   **トピック抽出:** 高度なクラスタリングによる話題の自動分類。
    *   **構造可視化:** 共起ネットワークによる単語間のつながりの把握。
    *   **AI 要約:** LLM を活用した文脈に沿った要約生成。
*   **次世代のインサイト抽出 (Core Pillars):**
    1.  **Delta分析 (コホート比較):** 属性や行動によるグループ間の「決定的な差異」を抽出し、成功要因や離脱原因を特定。
    2.  **アーキタイプ分類 (ペルソナ抽出):** 会話傾向からユーザーを性格的・目的的なタイプに分類し、具体的な像（ペルソナ）を可視化。
    3.  **Gap分析 (潜在ニーズ発見):** 表面化していないが文脈に含まれる潜在的な課題や機会を LLM で推論。
    4.  **モーメント分析 (熱量変化):** 会話中の「心が動いた瞬間（Aha! Moment）」や「離脱の兆候」を特定し、感情の質的な動きを捕捉。
*   **直感的なビジュアライゼーション:** インタラクティブなグラフにより、非専門家でも分析結果を即座に理解可能。
*   **双方向のプレゼンテーション層 (UX):**
    *   **Advanced Jupyter Widgets:** 分析対象の選択、期間指定、パラメータ調整をノートブック上で行えるインタラクティブな資料。
    *   **Professional Dashboard (Streamlit):** クライアントへのプレゼンテーションや共有に最適な、HTMLベースの本格的なダッシュボード実装。

## 🏗️ Architecture

```mermaid
graph TD
    subgraph Data_Sources
        NUCC[NUCC Corpus]
        CSV[Local Raw CSV]
        HF[HuggingFace (Optional)]
    end

    subgraph Analysis_Pipeline [src/analysis]
        Loader[Local/NUCC Loader] --> Pre[Preprocessor]
        Pre --> Engine1[Archetype Engine]
        Pre --> Engine2[Gap Analysis Engine]
        Pre --> Engine3[Delta Analysis]
    end

    subgraph Outputs
        NB[Jupyter Notebook]
        Plotly[User Maps & Insights]
    end

    NUCC --> Loader
    CSV --> Loader
    Engine1 --> Plotly
    Engine2 --> NB
    Engine3 --> NB
```

## 📂 Directory Structure

```text
Chat-Log-Analyzer/
├── data/                   # 分析用データ
│   └── raw/               # NUCCやCSVファイルを配置
├── notebooks/              # 分析ダッシュボード (Showcase)
│   ├── 01_overview.ipynb           # データ概要・Delta分析
│   ├── 02_archetype_analysis.ipynb # アーキタイプ抽出・可視化
│   └── 03_gap_analysis.ipynb       # 潜在ニーズ発見
├── src/                    # フレームワーク本体
│   ├── loader/            # NUCCLoader, LocalDatasetLoader
│   ├── preprocessor/      # Tokenizer
│   ├── analysis/          # Archetype, Gap, Delta Engines
│   └── visualization/     # Plotly Charts
├── tests/                  # 基本的なテスト
├── scripts/                # 検証用スクリプト (verify_*)
├── requirements.txt
└── README.md
```

## 🛠️ Installation & Setup

1.  **Clone & Enter**
    ```bash
    git clone <repository_url>
    cd Chat-Log-Analyzer
    ```

2.  **Environment Setup**
    ```bash
    # 仮想環境作成と依存ライブラリのインストール
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    pip install -r requirements.txt
    ```

3.  **Data Preparation**
    *   `data/raw/nucc/nucc` に名大会話コーパスのテキストファイルを配置してください。
    *   または `data/raw/` に分析したいCSVファイルを配置してください。

## 📊 Usage

主要な機能は Jupyter Notebook から呼び出します。

```python
from src.loader.nucc_loader import NUCCLoader
from src.analysis.archetype import ArchetypeEngine

# データの読み込み
loader = NUCCLoader(data_dir='data/raw/nucc/nucc')
df = loader.load()

# アーキタイプ分析の実行
engine = ArchetypeEngine(n_clusters=4)
features = engine.analyze_user_characteristics(df)
archetypes = engine.classify_archetypes(features)

# 結果の確認
display(archetypes.head())
```

## 📜 License

[MIT License](LICENSE)
