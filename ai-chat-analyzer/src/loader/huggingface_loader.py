# %% writefile Chat-Log-Analyzer/ai-chat-analyzer/src/loader/huggingface_loader.py
import pandas as pd
from datasets import load_dataset
from .base_loader import BaseLoader
import random

class HuggingFaceLoader(BaseLoader):
    def __init__(self, dataset_name="nu-dialogue/real-persona-chat", split="train"):
        """
        Args:
            dataset_name (str): HuggingFace dataset name.
            split (str): Split to load ('train', 'valid', etc.)
        """
        self.dataset_name = dataset_name
        self.split = split

    def load(self) -> pd.DataFrame:
        """
        Load dataset and convert to standard DataFrame schema.
        
        Returns:
            pd.DataFrame: DataFrame with columns:
                - timestamp (datetime)
                - session_id (str)
                - user_id (str)
                - speaker (str): 'user' or 'ai'
                - message (str)
                - user_age (str): Mocked attribute
                - user_gender (str): Mocked attribute
        """
        print(f"Loading dataset: {self.dataset_name} ({self.split})...")
        try:
            # Note: This dataset might require accepting terms on Hugging Face website.
            # If authentication error occurs, run `huggingface-cli login` in terminal.
            dataset = load_dataset(self.dataset_name, split=self.split, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please ensure you have accepted the dataset terms on Hugging Face and logged in.")
            return pd.DataFrame()

        records = []
        
        # ユーザー属性のモック用キャッシュ (同じユーザーIDには常に同じ属性を割り当てるため)
        user_attributes_cache = {}

        print("Processing records...")
        for row in dataset:
            dialogue_id = str(row.get("dialogue_id"))
            interlocutors = row.get("interlocutors", [])
            utterances = row.get("utterances", [])

            # 対話ペアの役割割り当て（開発用）
            # リストの0番目を 'user', 1番目を 'ai' と仮定する（またはその逆）
            if len(interlocutors) >= 2:
                role_map = {
                    interlocutors[0]: "user",
                    interlocutors[1]: "ai"
                }
            else:
                # 話者が一人の場合などはスキップまたは適当に割り当て
                continue

            for utt in utterances:
                interlocutor_id = utt.get("interlocutor_id")
                role = role_map.get(interlocutor_id, "unknown")
                
                # 'user' の場合のみ属性を付与（AI側の属性は分析対象外とする場合）
                age = ""
                gender = ""
                
                if role == "user":
                    if interlocutor_id not in user_attributes_cache:
                        # 開発用にランダムな属性を生成・固定
                        user_attributes_cache[interlocutor_id] = {
                            "age": random.choice(["10s", "20s", "30s", "40s", "50s", "60s+"]),
                            "gender": random.choice(["Male", "Female", "Other"])
                        }
                    attrs = user_attributes_cache[interlocutor_id]
                    age = attrs["age"]
                    gender = attrs["gender"]

                # タイムスタンプの処理 (データがない場合は現在時刻等で埋める)
                timestamp = utt.get("timestamp")
                if timestamp is None:
                    timestamp = pd.Timestamp.now()
                else:
                    try:
                        timestamp = pd.to_datetime(timestamp)
                    except:
                        timestamp = pd.Timestamp.now()

                records.append({
                    "timestamp": timestamp,
                    "session_id": dialogue_id,
                    "user_id": interlocutor_id,
                    "speaker": role,
                    "message": utt.get("text", ""),
                    "user_age": age,
                    "user_gender": gender
                })

        df = pd.DataFrame(records)
        
        # 型変換と整理
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # user_age, user_gender の欠損（AIの発話など）は NaN または適当な値で埋めるかそのまま
            # ここではAIの属性はNoneのままにします
        
        print(f"Successfully loaded {len(df)} records.")
        return df
