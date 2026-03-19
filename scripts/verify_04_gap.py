import sys
from pathlib import Path
import pandas as pd

# プロジェクトルートを追加
sys.path.append(str(Path.cwd()))

def verify_gap_analysis():
    print("=== Start Verification: Gap Analysis ===")
    
    # 1. NUCCデータのパス確認
    data_path = Path("data/raw/nucc/nucc")
    if not data_path.exists():
        print(f"[ERROR] NUCC data not found.")
        return
        
    try:
        from src.loader.nucc_loader import NUCCLoader
        from src.preprocessor.tokenizer import Tokenizer
        from src.analysis.archetype import ArchetypeEngine
        from src.analysis.gap_analysis import GapAnalysisEngine
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return

    # 2. データロード (軽量化のためファイル数制限などはLoaderにないので、ロード後にheadする)
    print("Loading data...")
    loader = NUCCLoader(data_dir=str(data_path))
    df = loader.load()
    
    if df.empty:
        print("[ERROR] DataFrame is empty.")
        return
        
    # テスト時間を短縮するためにサンプリング
    df_sample = df.head(1000).copy()
    print(f"Using {len(df_sample)} records for testing.")
    
    # 3. 前処理 & アーキタイプ (Gap分析の前提)
    print("Pre-processing...")
    tokenizer = Tokenizer()
    df_sample['tokenized_text'] = df_sample['text'].apply(lambda x: tokenizer.tokenize(x))
    
    arch_engine = ArchetypeEngine(n_clusters=3)
    features = arch_engine.analyze_user_characteristics(df_sample)
    labeled_df = arch_engine.classify_archetypes(features)
    df_merged = df_sample.merge(labeled_df[['archetype_label']], on='user_id', how='left')
    
    # 4. Gap Analysis実行
    print("Running Gap Analysis...")
    gap_engine = GapAnalysisEngine() # Default rule-based
    
    gaps = gap_engine.analyze_gaps(df_merged)
    
    if gaps.empty:
        print("[WARN] No gaps found (maybe sample size too small or no keywords matched).")
    else:
        print(f"[SUCCESS] Found {len(gaps)} gap signals.")
        print(gaps[['user_id', 'text', 'matched_keywords']].head().to_string())
        
        # 特定アーキタイプでのテスト
        target = df_merged['archetype_label'].iloc[0] # 存在するラベルを使う
        print(f"\nTesting target archetype: {target}")
        gaps_target = gap_engine.analyze_gaps(df_merged, target_archetype=target)
        print(f"Found {len(gaps_target)} gaps for {target}.")

if __name__ == "__main__":
    verify_gap_analysis()
