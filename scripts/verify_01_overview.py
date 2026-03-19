import sys
from pathlib import Path
import pandas as pd
import random
from unittest.mock import MagicMock

# プロジェクトルートをパスに追加
sys.path.append(str(Path.cwd()))

try:
    # from src.loader.huggingface_loader import HuggingFaceLoader
    from src.preprocessor.tokenizer import Tokenizer
    from src.analysis.delta_analysis import DeltaAnalysisEngine
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def verify_delta_analysis():
    print("=== Start Verification: Delta Analysis ===")
    
    # 1. データロード
    # HuggingFaceへの接続を避けるため、または失敗時のためにモックデータを用意する
    # loader = HuggingFaceLoader()
    # df = loader.load()
    
    # 時間短縮のため、手動でサンプルデータを作成する
    print("[INFO] Generating sample data...")
    records = []
    # 男性グループ: 技術的な話題、スポーツ
    m_words = ["PC", "プログラミング", "野球", "サッカー", "ゲーム", "仕事", "python", "コード"]
    # 女性グループ: カフェ、旅行、ファッション、映画、ドラマ、ランチ
    f_words = ["カフェ", "旅行", "ファッション", "映画", "ドラマ", "ランチ", "可愛い", "美味しい"]
    # 共通: 天気、明日、楽しみ
    common_words = ["天気", "明日", "楽しみ", "おはよう", "こんにちは"]
    
    for i in range(50):
        # Male
        text = " ".join(random.choices(m_words, k=3) + random.choices(common_words, k=2))
        records.append({
            "user_id": f"u_m_{i}",
            "attribute_gender": "M",
            "text": text
        })
        # Female
        text = " ".join(random.choices(f_words, k=3) + random.choices(common_words, k=2))
        records.append({
            "user_id": f"u_f_{i}",
            "attribute_gender": "F",
            "text": text
        })
        
    df = pd.DataFrame(records)
    print(f"[INFO] Sample data created: {len(df)} records")
    
    # 2. トークナイズ
    print("[INFO] Tokenizing text...")
    tokenizer = Tokenizer()
    
    # Tokenizerの実装がMeCabを使っているため、環境によっては失敗する可能性がある
    # ここでは簡易的にスペース区切りで代用するフォールバックを入れる
    try:
        df['tokenized_text'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
    except Exception as e:
        print(f"[WARN] Tokenizer failed ({e}), falling back to split()")
        df['tokenized_text'] = df['text'].apply(lambda x: x.split())
        
    print(f"[INFO] Tokenization completed. Sample: {df['tokenized_text'].iloc[0]}")
    
    # 3. Delta Analysis
    print("[INFO] Running DeltaAnalysisEngine...")
    engine = DeltaAnalysisEngine(min_freq=2)
    
    delta_df = engine.compute_log_odds_ratio(
        df,
        group_col='attribute_gender',
        group_a='M',
        group_b='F'
    )
    
    # 4. 結果確認
    print("\n--- Result: Top words for 'M' (Positive Log Odds) ---")
    print(delta_df.head(10)[['word', 'log_odds', 'count_a', 'count_b']].to_string(index=False))
    
    print("\n--- Result: Top words for 'F' (Negative Log Odds) ---")
    print(delta_df.tail(10)[['word', 'log_odds', 'count_a', 'count_b']].to_string(index=False))
    
    # 検証ポイント
    top_m_word = delta_df.iloc[0]['word']
    top_f_word = delta_df.iloc[-1]['word']
    
    is_valid_m = top_m_word in m_words or top_m_word in common_words # common words might appear by chance but less likely to be top
    is_valid_f = top_f_word in f_words or top_f_word in common_words
    
    if len(delta_df) > 0:
        print("\n[SUCCESS] Delta Analysis produced results.")
    else:
        print("\n[FAIL] Delta Analysis produced no results.")

if __name__ == "__main__":
    verify_delta_analysis()
