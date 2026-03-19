import sys
from pathlib import Path
import pandas as pd

# プロジェクトルートを追加
sys.path.append(str(Path.cwd()))

def verify_archetype_analysis():
    print("=== Start Verification: Archetype Analysis ===")
    
    # 1. ローカルデータの存在確認
    data_path = Path("data/raw/sample_chat.csv")
    if not data_path.exists():
        print(f"[ERROR] Data file not found at {data_path}")
        return
        
    print(f"[INFO] Using data: {data_path}")
    
    try:
        from src.loader.local_loader import LocalDatasetLoader
        from src.preprocessor.tokenizer import Tokenizer
        from src.analysis.archetype import ArchetypeEngine
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return

    # 2. データロード
    loader = LocalDatasetLoader(file_path=str(data_path))
    df = loader.load()
    
    if df.empty:
        print("[ERROR] DataFrame is empty.")
        return
        
    # 3. トークナイズ (簡易版フォールバック付き)
    print("[INFO] Tokenizing...")
    tokenizer = Tokenizer()
    try:
        df['tokenized_text'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
    except Exception as e:
        print(f"[WARN] Tokenizer failed ({e}), using split fallback")
        df['tokenized_text'] = df['text'].apply(lambda x: x.split())
        
    # 4. ArchetypeEngine 実行
    print("[INFO] Running ArchetypeEngine...")
    engine = ArchetypeEngine(n_clusters=3) # サンプルデータが少ないのでクラスタ数を減らす
    
    try:
        features_df = engine.analyze_user_characteristics(df)
        print("\n--- User Features (Head) ---")
        print(features_df.head().to_string())
        
        archetype_df = engine.classify_archetypes(features_df)
        print("\n--- Archetype Results (Head) ---")
        print(archetype_df[['msg_count', 'avg_sentiment', 'archetype_label']].head().to_string())
        
        # 検証ロジック
        if 'archetype_label' in archetype_df.columns and not archetype_df.empty:
            print("\n[SUCCESS] Archetype Analysis completed successfully.")
        else:
            print("\n[FAIL] Archetype Analysis failed to generate labels.")
            
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_archetype_analysis()
