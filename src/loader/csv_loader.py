import pandas as pd
import chardet
import os
from pathlib import Path
from typing import Optional, Dict, Any
from .base_loader import BaseLoader


class CSVLoader(BaseLoader):
    """CSVファイルからチャットデータを読み込むローダー"""
    
    # 推奨されるエンコーディングの優先順位
    ENCODING_PRIORITY = ['utf-8', 'utf-8-sig', 'shift-jis', 'cp932', 'latin-1']
    
    # 想定されるスキーマ
    EXPECTED_COLUMNS = {'user_id', 'message', 'timestamp', 'session_id'}
    REQUIRED_COLUMNS = {'user_id', 'message'}
    
    def __init__(self, file_path: str):
        """
        CSVLoaderを初期化
        
        Args:
            file_path (str): CSVファイルのパス
            
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: ファイルパスが無効な場合
        """
        self.file_path = file_path
        self._validate_file_path()
    
    def _validate_file_path(self) -> None:
        """ファイルパスの妥当性を検証"""
        if not self.file_path:
            raise ValueError("ファイルパスが空です")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {self.file_path}")
        
        if not self.file_path.lower().endswith('.csv'):
            raise ValueError(f"CSVファイルではありません: {self.file_path}")
        
        if not os.access(self.file_path, os.R_OK):
            raise PermissionError(f"ファイルを読み取る権限がありません: {self.file_path}")
    
    def _detect_encoding(self) -> str:
        """
        ファイルのエンコーディングを検出
        
        Returns:
            str: 検出されたエンコーディング
        """
        try:
            # chardectで最初の10KBを読み取ってエンコーディングを検出
            with open(self.file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                detected = result.get('encoding', 'utf-8')
                
                if detected is None:
                    return 'utf-8'
                
                return detected
        except Exception as e:
            print(f"エンコーディング検出に失敗しました: {e}")
            return 'utf-8'
    
    def _try_load_with_encoding(self, encoding: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        指定されたエンコーディングでCSVを読み込む試行
        
        Args:
            encoding (str): エンコーディング
            **kwargs: pd.read_csvに渡す追加引数
            
        Returns:
            Optional[pd.DataFrame]: 読み込まれたDataFrame、または失敗時はNone
        """
        try:
            df = pd.read_csv(self.file_path, encoding=encoding, **kwargs)
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            return None
    
    def _load_with_encoding_fallback(self, **kwargs) -> pd.DataFrame:
        """
        複数のエンコーディングで順次試行してCSVを読み込む
        
        Args:
            **kwargs: pd.read_csvに渡す追加引数
            
        Returns:
            pd.DataFrame: 読み込まれたDataFrame
            
        Raises:
            ValueError: どのエンコーディングでも読み込めない場合
        """
        # 検出されたエンコーディングを優先
        detected_encoding = self._detect_encoding()
        encodings_to_try = [detected_encoding] + self.ENCODING_PRIORITY
        
        # 重複を削除
        encodings_to_try = list(dict.fromkeys(encodings_to_try))
        
        for encoding in encodings_to_try:
            df = self._try_load_with_encoding(encoding, **kwargs)
            if df is not None:
                print(f"[OK] エンコーディング '{encoding}' で正常に読み込まれました")
                return df
        
        raise ValueError(
            f"CSVファイルを読み込めません。試行したエンコーディング: {', '.join(encodings_to_try)}"
        )
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        DataFrameのスキーマを検証
        
        Args:
            df (pd.DataFrame): 検証するDataFrame
            
        Raises:
            ValueError: 必須カラムが不足している場合
        """
        columns = set(df.columns)
        
        # 必須カラムの確認
        missing_required = self.REQUIRED_COLUMNS - columns
        if missing_required:
            raise ValueError(
                f"必須カラムが不足しています: {', '.join(missing_required)}"
            )
        
        # 推奨カラムの確認（警告のみ）
        missing_recommended = self.EXPECTED_COLUMNS - columns
        if missing_recommended:
            print(f"[WARN] 推奨カラムが不足しています: {', '.join(missing_recommended)}")
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データ型を適切に変換
        
        Args:
            df (pd.DataFrame): 変換するDataFrame
            
        Returns:
            pd.DataFrame: 型変換されたDataFrame
        """
        df = df.copy()
        
        # user_id を string に（NaNも保持）
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].astype('object')  # NaNを保持
        
        # message を string に（NaNも保持）
        if 'message' in df.columns:
            df['message'] = df['message'].astype('object')  # NaNを保持
        
        # session_id が存在すれば string に
        if 'session_id' in df.columns:
            df['session_id'] = df['session_id'].astype(str)
        
        # timestamp が存在すれば datetime に試行
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except (ValueError, TypeError):
                print(f"[WARN] 'timestamp' を datetime に変換できませんでした")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値を処理
        
        Args:
            df (pd.DataFrame): 処理するDataFrame
            
        Returns:
            pd.DataFrame: 欠損値が処理されたDataFrame
        """
        df = df.copy()
        
        # 'message' カラムの欠損行を削除（NaNまたは空文字列）
        if 'message' in df.columns:
            initial_rows = len(df)
            # NaN または 空文字列または空白のみの行を削除
            mask = df['message'].notna() & (df['message'].astype(str).str.strip() != '')
            df = df[mask]
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                print(f"[WARN] {dropped_rows} 行の欠損/空白 'message' を削除しました")
        
        # 'user_id' カラムの欠損値を埋める（NaN）
        if 'user_id' in df.columns:
            null_count = df['user_id'].isna().sum()
            if null_count > 0:
                df.loc[df['user_id'].isna(), 'user_id'] = 'unknown_user'
                print(f"[WARN] {null_count} 個の欠損 'user_id' を 'unknown_user' で埋めました")
        
        return df
    
    def load(self, **kwargs) -> pd.DataFrame:
        """
        CSVファイルを読み込む
        
        Args:
            **kwargs: pd.read_csvに渡す追加引数
            
        Returns:
            pd.DataFrame: 読み込まれたDataFrame
            
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: ファイルが読み込めない、またはスキーマが無効な場合
            PermissionError: ファイルを読む権限がない場合
        """
        try:
            # エンコーディングフォールバック付きで読み込み
            df = self._load_with_encoding_fallback(**kwargs)
            
            # スキーマ検証
            self._validate_schema(df)
            
            # 欠損値処理（型変換前）
            df = self._handle_missing_values(df)
            
            # 型変換
            df = self._convert_types(df)
            
            print(f"[OK] {len(df)} 行のデータを読み込みました")
            return df
            
        except Exception as e:
            raise ValueError(f"CSVファイルの読み込みに失敗しました: {e}")