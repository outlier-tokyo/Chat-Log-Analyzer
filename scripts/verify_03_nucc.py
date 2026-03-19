import sys
from pathlib import Path
import pandas as pd

# プロジェクトルートを追加
sys.path.append(str(Path.cwd()))

def verify_nucc_loader():
    print("=== Start Verification: NUCC Loader ===")
    
    # 1. NUCCデータのパス確認
    data_path = Path("data/raw/nucc/nucc")
    if not data_path.exists():
        print(f"[ERROR] NUCC data directory not found at {data_path}")
        return
        
    print(f"[INFO] Using data directory: {data_path}")
    
    try:
        from src.loader.nucc_loader import NUCCLoader
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return

    # 2. データロード
    loader = NUCCLoader(data_dir=str(data_path))
    try:
        # デモ用に1ファイルだけロードするテスト用メソッドがあればよいが、
        # 現状は全ロードなので、とりあえず実行してみる（ファイル数が多いので注意）
        # 簡易的に最初の5ファイルだけに制限するハックを入れるか、そのまま実行するか。
        # NUCCLoader側にファイル数制限機能がないので、テストのために一時的にファイルをリストアップして制限するロジックをここでエミュレートは難しい（Loader内部実装の問題）。
        # しかしNUCCはテキストファイルなので数千件程度なら数秒で終わるはず。
        
        df = loader.load()
        
        if df.empty:
            print("[ERROR] DataFrame is empty.")
        else:
            print(f"[SUCCESS] Loaded {len(df)} records.")
            print(f"Columns: {df.columns.tolist()}")
            print("\n--- Sample Data ---")
            print(df[['timestamp', 'session_id', 'user_id', 'text', 'attribute_gender']].head().to_string())
            
            print("\n--- Text Content Debug ---")
            sample_text = df['text'].iloc[0]
            # Write to file to verify content (bypassing terminal encoding issues)
            with open("scripts/debug_output.txt", "w", encoding="utf-8") as f:
                f.write(sample_text)
            print("Wrote sample text to scripts/debug_output.txt")
            
            # 必須カラムのチェック
            required_cols = ['timestamp', 'user_id', 'text']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"[ERROR] Missing required columns: {missing}")
            else:
                print("[INFO] All required columns are present.")

    except Exception as e:
        print(f"[ERROR] Loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_nucc_loader()
