import pandas as pd
from pathlib import Path
from .base_loader import BaseLoader

class LocalDatasetLoader(BaseLoader):
    def __init__(self, file_path: str):
        """
        ローカルのデータセットファイルを読み込むローダー
        
        Args:
            file_path (str): 読み込むファイルのパス (CSV, JSON, Parquet等)
        """
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        """
        Load dataset from local file.
        Detects format by extension.
        """
        print(f"Loading local dataset: {self.file_path}...")
        
        if not self.file_path.exists():
            print(f"[ERROR] File not found: {self.file_path}")
            return pd.DataFrame()
            
        try:
            if self.file_path.suffix == '.csv':
                df = pd.read_csv(self.file_path)
            elif self.file_path.suffix == '.json':
                df = pd.read_json(self.file_path)
            elif self.file_path.suffix == '.parquet':
                df = pd.read_parquet(self.file_path)
            else:
                print(f"[ERROR] Unsupported file format: {self.file_path.suffix}")
                return pd.DataFrame()
                
            # カラム名の標準化（必要に応じてマッピング）
            # 最低限必要なカラム: timestamp, user_id, text (or message)
            if 'message' not in df.columns and 'text' in df.columns:
                df['message'] = df['text']
            elif 'message' not in df.columns:
                print("[WARN] 'message' or 'text' column missing. Analysis might fail.")
                
            if 'timestamp' not in df.columns:
                print("[WARN] 'timestamp' missing. Filling with dummy dates.")
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            if 'user_id' not in df.columns:
                df['user_id'] = 'unknown_user'

            print(f"Successfully loaded {len(df)} records from local file.")
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load local file: {e}")
            return pd.DataFrame()
