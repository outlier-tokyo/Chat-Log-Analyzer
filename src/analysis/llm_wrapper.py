import os
from openai import OpenAI
from typing import List

class LLMSummarizer:
    """LLM を使用した会話要約クラス"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        LLMSummarizer を初期化
        
        Args:
            api_key (str): OpenAI API Key. None の場合は環境変数から取得。
            model (str): 使用するモデル名
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("[WARN] OpenAI API Key is not set. Summarization will only return placeholders.")
        
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model

    def summarize(self, text: str, max_words: int = 150) -> str:
        """
        与えられたテキストを要約する
        
        Args:
            text (str): 要約対象のテキスト
            max_words (int): 要約の最大単語数/文字数の目安
            
        Returns:
            str: 要約結果
        """
        if not self.client:
            return "Summarization skipped: No API Key provided."

        if not text or len(text.strip()) == 0:
            return "No text provided for summarization."

        system_prompt = (
            "あなたは高度なデータアナリストです。与えられた会話ログの内容を分析し、"
            f"主要なトピック、ユーザーの意図、および重要なインサイトを {max_words} 字程度で簡潔に要約してください。"
            "客観的かつプロフェッショナルなトーンで記述してください。"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"以下のログを要約してください:\n\n{text}"}
                ],
                temperature=0.5,
                max_tokens=500
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            return f"Summarization error: {str(e)}"

    def summarize_clusters(self, df: 'pd.DataFrame', cluster_id: int, top_n: int = 20) -> str:
        """
        特定のクラスタに属するテキスト群を要約する
        
        Args:
            df (pd.DataFrame): 'text' と 'cluster' カラムを持つ DataFrame
            cluster_id (int): 要約対象のクラスタ番号
            top_n (int): 要約に使用する代表的なメッセージ数
            
        Returns:
            str: クラスタの要約
        """
        cluster_texts = df[df['cluster'] == cluster_id]['text'].head(top_n).tolist()
        combined_text = "\n".join([f"- {t}" for t in cluster_texts])
        
        prompt = f"このクラスタ（Cluster {cluster_id}）には以下のような発話が含まれています。この話題の核心を1文で表現してください。"
        
        return self.summarize(f"{prompt}\n\n{combined_text}", max_words=50)