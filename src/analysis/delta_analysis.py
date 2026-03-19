import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
import math

class DeltaAnalysisEngine:
    """
    コホート（グループ）間の差異を分析するエンジン
    """
    
    def __init__(self, min_freq: int = 5):
        """
        DeltaAnalysisEngine を初期化
        
        Args:
            min_freq (int): 分析対象とする単語の最小出現頻度
        """
        self.min_freq = min_freq

    def compute_log_odds_ratio(
        self, 
        df: pd.DataFrame, 
        group_col: str, 
        group_a: str, 
        group_b: str, 
        text_col: str = 'tokenized_text'
    ) -> pd.DataFrame:
        """
        2つのグループ間で出現頻度の差（Log Odds Ratio）を計算する。
        正の値は group_a で特徴的、負の値は group_b で特徴的。
        
        Args:
            df (pd.DataFrame): データの DataFrame
            group_col (str): グループ分けに使用するカラム名
            group_a (str): グループAのラベル
            group_b (str): グループBのラベル
            text_col (str): 単語リスト（List[str]）が入っているカラム名
            
        Returns:
            pd.DataFrame: word, log_odds, count_a, count_b を含む結果
        """
        # 各グループの単語頻度をカウント
        tokens_a = [token for sublist in df[df[group_col] == group_a][text_col] for token in sublist]
        tokens_b = [token for sublist in df[df[group_col] == group_b][text_col] for token in sublist]
        
        counts_a = Counter(tokens_a)
        counts_b = Counter(tokens_b)
        
        all_words = set(list(counts_a.keys()) + list(counts_b.keys()))
        n_a = sum(counts_a.values())
        n_b = sum(counts_b.values())
        
        results = []
        for word in all_words:
            c_a = counts_a.get(word, 0)
            c_b = counts_b.get(word, 0)
            
            # 最小頻度チェック
            if (c_a + c_b) < self.min_freq:
                continue
                
            # Log Odds Ratio with Dirichlet Prior (Smoothing)
            # 参照: https://en.wikipedia.org/wiki/Odds_ratio#Weighted_log_odds_ratio
            # 自然対数を用いて計算
            odds_a = (c_a + 0.5) / (n_a - c_a + 0.5)
            odds_b = (c_b + 0.5) / (n_b - c_b + 0.5)
            log_odds = math.log(odds_a / odds_b)
            
            results.append({
                'word': word,
                'log_odds': log_odds,
                'count_a': c_a,
                'count_b': c_b,
                'total': c_a + c_b
            })
            
        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df = res_df.sort_values('log_odds', ascending=False)
            
        return res_df

    def analyze_topic_shift(self, df: pd.DataFrame, group_col: str, cluster_col: str = 'cluster') -> pd.DataFrame:
        """
        グループごとのトピック分布の差異を計算する
        """
        pivot = df.pivot_table(
            index=group_col, 
            columns=cluster_col, 
            aggfunc='size', 
            fill_value=0
        )
        # 割合に変換
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        return pivot_pct
