import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

class ArchetypeEngine:
    """
    ユーザーの会話行動を分析し、アーキタイプ（ペルソナ）に分類するエンジン
    """
    
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """
        Args:
            n_clusters (int): 分類するアーキタイプの数
            random_state (int): 乱数シード
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.cluster_labels = None
        
        # 簡易感情辞書 (ポジティブ/ネガティブ)
        # 本来は外部辞書を使うべきだが、依存関係を減らすため内蔵する
        # TODO: 本番運用時は、以下のいずれかの方法で感情スコアの精度を向上させること
        # 1. 'osier/japanese-sentiment-polarity' などの外部辞書をロードする
        # 2. HuggingFaceの感情分析モデル (e.g. 'koheiduck/bert-japanese-finetuned-sentiment') を使用する
        # 3. Cloud Natural Language API 等の外部APIを利用する
        self.pos_words = set([
            "良い", "好き", "楽しい", "最高", "感謝", "嬉しい", "面白い", "すごい", 
            "便利", "親切", "丁寧", "安心", "満足", "期待", "素晴らしい", "good", "great", "thanks"
        ])
        self.neg_words = set([
            "悪い", "嫌い", "つまらない", "最悪", "不満", "悲しい", "難しい", "ひどい", 
            "不便", "遅い", "高い", "複雑", "不安", "失敗", "残念", "bad", "hate", "slow"
        ])

    def _calculate_sentiment(self, text: str, tokenized_text: List[str]) -> float:
        """
        簡易的な感情スコアを計算 (-1.0 to 1.0)
        TODO: プロダクション環境では、BERT等のBERTモデルを用いた推論に置き換えることを強く推奨
        """
        if not tokenized_text:
            return 0.0
            
        score = 0
        count = 0
        for token in tokenized_text:
            if token in self.pos_words:
                score += 1
                count += 1
            elif token in self.neg_words:
                score -= 1
                count += 1
                
        if count == 0:
            return 0.0
        return score / count

    def analyze_user_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ユーザーごとの特徴量を算出する
        
        Args:
            df (pd.DataFrame): 'user_id', 'text', 'tokenized_text' カラムが必要
            
        Returns:
            pd.DataFrame: user_id をインデックスとし、特徴量をカラムに持つDF
        """
        # 1. 基本統計量の集計
        # 1. 基本統計量の集計
        agg_dict = {'text': 'count'}
        
        # session_id が存在する場合のみ集計対象に含める
        if 'session_id' in df.columns:
            agg_dict['session_id'] = 'nunique'
            
        user_stats = df.groupby('user_id').agg(agg_dict)
        
        # カラム名のリネーム (存在するものだけ)
        rename_map = {'text': 'msg_count'}
        if 'session_id' in df.columns:
            rename_map['session_id'] = 'session_count'
            
        user_stats = user_stats.rename(columns=rename_map)
        
        # セッションIDがない場合のフォールバック（全員1セッション扱い）
        if 'session_count' not in user_stats.columns:
             user_stats['session_count'] = 1
             
        # 2. テキスト特徴量の算出 (平均文字数、感情スコア)
        # 注意: applyは遅いので、データ量が多い場合はベクトル化を検討すべき
        df['char_length'] = df['text'].astype(str).str.len()
        df['sentiment'] = df.apply(
            lambda x: self._calculate_sentiment(x['text'], x.get('tokenized_text', [])), 
            axis=1
        )
        
        text_stats = df.groupby('user_id').agg({
            'char_length': 'mean',
            'sentiment': 'mean'
        }).rename(columns={'char_length': 'avg_char_len', 'sentiment': 'avg_sentiment'})
        
        # 3. 語彙多様性 (Type-Token Ratio) の算出
        # 全発話を結合して計算するのは重いので、トークンリストから推定
        def calculate_ttr(tokens_series):
            all_tokens = []
            for tokens in tokens_series:
                if isinstance(tokens, list):
                    all_tokens.extend(tokens)
            if not all_tokens:
                return 0.0
            return len(set(all_tokens)) / len(all_tokens)

        if 'tokenized_text' in df.columns:
            ttr_stats = df.groupby('user_id')['tokenized_text'].apply(calculate_ttr)
            ttr_stats.name = 'vocabulary_richness'
        else:
            ttr_stats = pd.Series(0.0, index=user_stats.index, name='vocabulary_richness')

        # 結合
        features_df = pd.concat([user_stats, text_stats, ttr_stats], axis=1).fillna(0)
        return features_df

    def classify_archetypes(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量に基づいてユーザーをクラスタリングする
        
        Args:
            features_df (pd.DataFrame): compute_user_characteristics の出力
            
        Returns:
            pd.DataFrame: features_df に 'archetype_id', 'archetype_label' を追加したもの
        """
        # クラスタリング用の特徴量を選択
        # msg_count, avg_char_len, avg_sentiment, vocabulary_richness
        X = features_df[['msg_count', 'avg_char_len', 'avg_sentiment', 'vocabulary_richness']].copy()
        
        # スケーリング (K-Meansは距離ベースなので必須)
        X_scaled = self.scaler.fit_transform(X)
        
        # K-Means 実行
        # データ数が少ない場合は警告を出さないように注意
        n_samples = len(X)
        k = min(self.n_clusters, n_samples)
        if k < 2:
            self.cluster_labels = np.zeros(n_samples) # データが少なすぎる場合は全て0
        else:
            self.model = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto')
            self.cluster_labels = self.model.fit_predict(X_scaled)
            
        features_df['archetype_id'] = self.cluster_labels
        
        # クラスタの意味づけ（ラベル生成）
        # 重心データを見ながらルールベースで名前を付ける
        if k >= 2:
            labels = self._generate_cluster_labels(features_df)
            features_df['archetype_label'] = features_df['archetype_id'].map(labels)
        else:
            features_df['archetype_label'] = 'General User'
            
        return features_df

    def _generate_cluster_labels(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        各クラスタの特徴を見て自動的にラベルを生成する
        """
        labels = {}
        means = df.groupby('archetype_id')[['msg_count', 'avg_sentiment', 'vocabulary_richness']].mean()
        
        # 全体平均と比較
        global_means = df[['msg_count', 'avg_sentiment', 'vocabulary_richness']].mean()
        
        for cluster_id in means.index:
            row = means.loc[cluster_id]
            
            label_parts = []
            
            # 発話量による判定
            if row['msg_count'] > global_means['msg_count'] * 1.5:
                label_parts.append("Active")
            elif row['msg_count'] < global_means['msg_count'] * 0.6:
                label_parts.append("Passive")
                
            # 感情による判定
            if row['avg_sentiment'] > 0.05: # 少しプラス
                label_parts.append("Positive")
            elif row['avg_sentiment'] < -0.05: # 少しマイナス
                label_parts.append("Critical")
                
            # 語彙力による判定（補足的）
            if row['vocabulary_richness'] > global_means['vocabulary_richness'] * 1.2:
                if not label_parts: label_parts.append("Articulate")
            
            # 組み合わせで命名
            if not label_parts:
                final_label = "Standard User"
            else:
                final_label = " ".join(label_parts) + " User"
                
            # 特別ルールの適用（よりキャッチーな名前に）
            if "Active Positive" in final_label:
                final_label = "Loyal Fan (Evangelist)"
            elif "Active Critical" in final_label:
                final_label = "Vocal Critic (Advisor)"
            elif "Passive Critical" in final_label:
                final_label = "At-Risk User"
            elif "Passive Positive" in final_label:
                final_label = "Silent Supporter"
                
            labels[cluster_id] = final_label
            
        return labels
