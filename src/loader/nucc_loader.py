import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from .base_loader import BaseLoader

class NUCCLoader(BaseLoader):
    """
    名古屋大学会話コーパス (NUCC) を読み込むためのローダー
    """
    def __init__(self, data_dir: str = "data/raw/nucc/nucc"):
        self.data_dir = Path(data_dir)

    def load(self) -> pd.DataFrame:
        """
        NUCCの全テキストファイルを読み込み、DataFrame化する
        """
        print(f"Loading NUCC dataset from: {self.data_dir}...")
        
        if not self.data_dir.exists():
            print(f"[ERROR] Directory not found: {self.data_dir}")
            return pd.DataFrame()

        all_records = []
        files = list(self.data_dir.glob("data*.txt"))
        
        if not files:
            print(f"[WARN] No data files found in {self.data_dir}")
            return pd.DataFrame()

        for file_path in files:
            records = self._parse_file(file_path)
            all_records.extend(records)

        df = pd.DataFrame(all_records)
        print(f"Successfully loaded {len(df)} utterances from {len(files)} files.")
        return df

    def _parse_file(self, file_path: Path) -> List[Dict]:
        """
        1つの会話ファイルをパースする
        ヘッダー情報（参加者属性など）も抽出して各発話に付与する
        """
        records = []
        filename = file_path.name
        
        # セッションID (ファイル名から抽出: data001.txt -> 001)
        session_id = filename.replace("data", "").replace(".txt", "")
        
        # デフォルト日付（ファイル内に記載がなければこれを使う）
        base_date = datetime(2000, 1, 1) 
        
        # 参加者情報のマッピング
        participants = {}
        
        current_speaker = None
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[WARN] Failed to read {filename}: {e}")
            return []

        # 1. ヘッダー情報の解析
        header_lines = [l for l in lines if l.startswith('＠')]
        dialogue_lines = [l for l in lines if not l.startswith('＠')]
        
        for line in header_lines:
            line = line.strip()
            # 日付抽出 ＠収集年月日：２００１年１０月１６日
            if "収集年月日" in line:
                date_str = line.split("：")[-1]
                # 全角数字を半角に変換などの処理が必要だが、簡易的にスキップまたは変換
                # ここでは簡易実装として固定日付または変換を試みる
                pass
            
            # 参加者抽出 ＠参加者F107：女性３０代後半、愛知県幡豆郡出身、愛知県幡豆郡在住
            # 形式: ＠参加者ID：属性
            match = re.search(r"＠参加者(\w+)：(.+)", line)
            if match:
                pid = match.group(1)
                attr_text = match.group(2)
                
                # 年代、性別抽出
                gender = "F" if "女性" in attr_text else "M" if "男性" in attr_text else "Unknown"
                
                # 年代 (e.g., ３０代後半 -> 30s)
                age = "Unknown"
                if "１０代" in attr_text: age = "10s"
                elif "２０代" in attr_text: age = "20s"
                elif "３０代" in attr_text: age = "30s"
                elif "４０代" in attr_text: age = "40s"
                elif "５０代" in attr_text: age = "50s"
                elif "６０代" in attr_text: age = "60s"
                
                participants[pid] = {
                    "gender": gender,
                    "age": age,
                    "raw_attr": attr_text
                }
        
        # 2. 会話ログの解析
        # 形式:
        # F107：＊＊＊の町というのは...
        # F023：１時間かからないぐらいだね。
        # 継続行（話者なしで行頭インデントなしの場合もあるが、NUCCは話者IDで始まることが多い）
        
        time_offset = 0 # 分単位で擬似的に時間を進める
        
        for line in dialogue_lines:
            line = line.strip()
            if not line:
                continue
                
            # 話者の切り替わり判定 (例: "F107：")
            # 全角コロン注意
            match = re.match(r"^([A-Z0-9]+?)：(.*)", line)
            if match:
                current_speaker = match.group(1)
                text = match.group(2)
            else:
                # 継続行とみなす
                if current_speaker:
                    text = line
                else:
                    continue # 話者不明の行はスキップ
            
            # ノイズ除去（注釈など ＜笑い＞ （うん））
            #text = re.sub(r"＜.*?＞", "", text)
            #text = re.sub(r"（.*?）", "", text)
            
            if not text.strip():
                continue

            # ユーザー属性の取得
            user_info = participants.get(current_speaker, {"gender": "Unknown", "age": "Unknown"})
            
            # タイムスタンプ生成 (擬似的に少しずつ進める)
            timestamp = base_date + timedelta(minutes=time_offset)
            time_offset += 1 # 1発話1分と仮定（分析用）
            
            records.append({
                "timestamp": timestamp,
                "session_id": session_id,
                "user_id": current_speaker,
                "text": text,
                "attribute_gender": user_info["gender"],
                "attribute_age": user_info["age"],
                "data_source": "NUCC"
            })
            
        return records
