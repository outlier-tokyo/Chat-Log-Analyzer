import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional, Dict
import warnings
import os


class TextVectorizer:
    """
    Sentence-BERTを使用したテキストベクトル化クラス
    セキュリティ考慮: ローカルのみでの処理、インターネットアップロードなし
    """
    
    # 推奨モデル
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_DIM = 768  # multilingual-mpnetの埋め込み次元
    
    def __init__(self, model_name: str = None, device: str = "cpu", use_mock: bool = False):
        """
        TextVectorizerを初期化
        
        Args:
            model_name (str): 使用するSentence-Transformerモデル名
            device (str): 実行デバイス ("cpu" のみ、セキュリティのためGPU無効)
            use_mock (bool): テスト用モック実装を使用するか
        """
        if model_name is None:
            model_name = self.DEFAULT_MODEL
        
        self.model_name = model_name
        self.device = device
        self.use_mock = use_mock
        self.embedding_dim = self.EMBEDDING_DIM
        
        # セキュリティ: CPU のみ、ローカルプロセッシング
        if device != "cpu":
            warnings.warn("Security: GPU disabled. CPU-only processing for confidential data.", UserWarning)
            self.device = "cpu"
        
        # Sentence-Transformerの遅延ロード
        self.model = None
        if not use_mock:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name, device=device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            except ImportError:
                warnings.warn("sentence-transformers not available. Using mock implementation.", UserWarning)
                self.use_mock = True
        
        print(f"[OK] TextVectorizer initialized: {model_name}")
        print(f"[OK] Embedding dimension: {self.embedding_dim}")
        print(f"[OK] Device: {self.device}")
        if self.use_mock:
            print(f"[WARN] Using mock implementation for testing")
    
    def encode(
        self,
        texts: Union[List[str], str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        テキストをベクトル化
        
        Args:
            texts (Union[List[str], str]): テキストまたはテキストリスト
            batch_size (int): バッチサイズ（メモリ効率性と速度のバランス）
            show_progress (bool): プログレスバー表示
            normalize (bool): ベクトル正規化（コサイン距離計算用）
            
        Returns:
            np.ndarray: 形状 (N, embedding_dim) のベクトル配列
        """
        # 単一テキストをリストに変換
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts or len(texts) == 0:
            raise ValueError("テキストが空です")
        
        try:
            if self.use_mock:
                # モック実装：テキストの長さとハッシュ値に基づいてランダムベクトルを生成
                embeddings = self._generate_mock_embeddings(texts, normalize)
            else:
                # Sentence-Transformersでベクトル化
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize
                )
            
            return embeddings
            
        except Exception as e:
            raise ValueError(f"ベクトル化に失敗しました: {e}")
    
    def _generate_mock_embeddings(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """モック実装：決定的にベクトルを生成（テスト用）"""
        np.random.seed(42)  # 再現性のため固定シード
        embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        
        # テキストのハッシュ値で微調整（同じテキストは同じベクトル）
        for i, text in enumerate(texts):
            np.random.seed(hash(text) % 2**32)
            embeddings[i] = np.random.randn(self.embedding_dim).astype(np.float32)
        
        if normalize:
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        return embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        複数テキストをバッチ処理でベクトル化（大規模データ向け）
        
        Args:
            texts (List[str]): テキストリスト
            batch_size (int): バッチサイズ
            show_progress (bool): プログレスバー表示
            
        Returns:
            np.ndarray: ベクトル配列
        """
        return self.encode(texts, batch_size=batch_size, show_progress=show_progress)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        2つのテキストのコサイン類似度を計算
        
        Args:
            text1 (str): テキスト1
            text2 (str): テキスト2
            
        Returns:
            float: コサイン類似度 (-1 ～ 1)
        """
        embeddings = self.encode([text1, text2], normalize=True)
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
    
    def similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        テキストリスト間のコサイン類似度マトリックスを計算
        
        Args:
            texts (List[str]): テキストリスト
            
        Returns:
            np.ndarray: 形状 (N, N) の類似度マトリックス
        """
        embeddings = self.encode(texts, normalize=True)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return similarity_matrix
    
    def find_similar_texts(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        クエリテキストに最も類似したテキストをtop_k件返す
        
        Args:
            query_text (str): クエリテキスト
            candidate_texts (List[str]): 候補テキストリスト
            top_k (int): 返す件数
            
        Returns:
            List[Tuple[str, float]]: (テキスト, 類似度) のリスト（降順）
        """
        # クエリのベクトル化
        query_embedding = self.encode(query_text, normalize=True)
        
        # 候補テキストのベクトル化
        candidate_embeddings = self.encode(candidate_texts, normalize=True)
        
        # コサイン類似度を計算
        similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
        
        # top_k件を取得
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [
            (candidate_texts[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def to_dataframe(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        テキストとベクトルをDataFrameに変換
        
        Args:
            texts (List[str]): テキストリスト
            embeddings (Optional[np.ndarray]): 既にベクトル化済みの場合
            
        Returns:
            pd.DataFrame: text とベクトル成分のカラムを含むDataFrame
        """
        if embeddings is None:
            embeddings = self.encode(texts)
        
        # DataFrameを作成
        data = {
            'text': texts,
            **{f'embedding_{i}': embeddings[:, i] for i in range(self.embedding_dim)}
        }
        
        df = pd.DataFrame(data)
        return df
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        method: str = "umap"
    ) -> np.ndarray:
        """
        ベクトルの次元削減（可視化用）
        
        Args:
            embeddings (np.ndarray): ベクトル配列
            n_components (int): 削減後の次元数
            method (str): 次元削減方法 ("umap" または "pca")
            
        Returns:
            np.ndarray: 次元削減されたベクトル
        """
        if method == "pca":
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_components)
                reduced = pca.fit_transform(embeddings)
                print(f"[OK] PCA: {self.embedding_dim}D -> {n_components}D")
                print(f"[OK] 説明分散比: {sum(pca.explained_variance_ratio_):.2%}")
                return reduced
            except ImportError:
                raise ImportError("scikit-learn がインストールされていません")
        
        elif method == "umap":
            try:
                import umap
                umap_model = umap.UMAP(n_components=n_components)
                reduced = umap_model.fit_transform(embeddings)
                print(f"[OK] UMAP: {self.embedding_dim}D -> {n_components}D")
                return reduced
            except ImportError:
                raise ImportError("umap-learn がインストールされていません")
        
        else:
            raise ValueError(f"不明な方法: {method}")
    
    def get_model_info(self) -> Dict:
        """
        モデル情報を取得
        
        Returns:
            Dict: モデル情報（名前、次元数、デバイス等）
        """
        max_seq_length = None
        if self.model is not None:
            max_seq_length = self.model.max_seq_length
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_seq_length": max_seq_length,
            "use_mock": self.use_mock
        }
    
    def __repr__(self) -> str:
        return (f"TextVectorizer(model='{self.model_name}', "
                f"dim={self.embedding_dim}, device='{self.device}')")