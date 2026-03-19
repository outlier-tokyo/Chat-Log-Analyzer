import pandas as pd
from typing import List, Dict, Optional
import random

class GapAnalysisEngine:
    """
    ユーザーの会話から「満たされていないニーズ (Gap)」や「潜在的な要望」を抽出するエンジン。
    LLM (Large Language Model) を活用して深い洞察を得ることを想定しているが、
    現在はキーワードベースのヒューリスティック分析も併用する。
    """
    
    def __init__(self, use_llm: bool = False, model_name: str = "local-rule-based"):
        """
        Args:
            use_llm (bool): LLMを使用するかどうか
            model_name (str): 使用するモデル名 ('bitnet', 'gpt-4o', etc.)
        """
        self.use_llm = use_llm
        self.model_name = model_name
        
        # ギャップを示唆するキーワード（日本語）
        self.gap_keywords = [
            "欲しい", "あればいいのに", "困る", "使いにくい", "不便", "できない", 
            "わからない", "面倒", "遅い", "高い", "足りない", "もっと", "改善",
            "want", "wish", "difficult", "hard to", "boring"
        ]

    def analyze_gaps(self, df: pd.DataFrame, target_archetype: str = None) -> pd.DataFrame:
        """
        ギャップ分析を実行する。
        
        Args:
            df (pd.DataFrame): 会話ログ
            target_archetype (str): 分析対象とするアーキタイプ（Noneの場合は全員）
            
        Returns:
            pd.DataFrame: 抽出されたインサイト情報
        """
        # 分析対象のフィルタリング
        if target_archetype and 'archetype_label' in df.columns:
            target_df = df[df['archetype_label'] == target_archetype].copy()
        else:
            target_df = df.copy()
            
        if target_df.empty:
            return pd.DataFrame()

        # 1. ルールベース抽出 (高速、コストゼロ)
        gap_records = self._extract_by_keywords(target_df)
        
        # 2. LLMによる深化 (オプション)
        if self.use_llm:
            # TODO: LLM連携 (BitNetやOpenAI API) を実装
            # 現在はシミュレーション
            llm_records = self._simulate_llm_insight(gap_records)
            return pd.DataFrame(llm_records)
        
        return pd.DataFrame(gap_records)

    def _extract_by_keywords(self, df: pd.DataFrame) -> List[Dict]:
        """
        キーワードマッチングによる抽出
        """
        results = []
        for _, row in df.iterrows():
            text = str(row['text'])
            matches = [kw for kw in self.gap_keywords if kw in text]
            if matches:
                results.append({
                    "timestamp": row['timestamp'],
                    "user_id": row['user_id'],
                    "text": text,
                    "matched_keywords": matches,
                    "insight_type": "Explicit Complaint/Wish",
                    "confidence": "Medium"
                })
        return results

    def _simulate_llm_insight(self, records: List[Dict]) -> List[Dict]:
        """
        LLMの動作をシミュレート（デモ用）
        """
        # 既存のレコードに「AIによる解釈」を追加するイメージ
        enhanced_records = []
        for r in records[:5]: # 全部やると重いので最初の5件だけ
            new_r = r.copy()
            new_r['ai_analysis'] = f"User seems frustrated about '{r['matched_keywords'][0]}'. suggested solution: Review UX."
            enhanced_records.append(new_r)
        return enhanced_records
