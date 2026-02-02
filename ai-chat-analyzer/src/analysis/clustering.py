"""
TopicClusteringクラス

テキストベクトルを使用してトピッククラスタリングを行う
- HDBSCAN クラスタリング（密度ベース、ノイズ検出対応）
- クラスタの要約と統計
- クラスタの可視化対応

セキュリティ: ローカルのみ処理
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("hdbscan not installed. Install with: pip install hdbscan")


class TopicClusterer:
    """トピッククラスタリング（HDBSCAN ベース）"""
    
    def __init__(self, min_cluster_size: int = 10, min_samples: int = 5, metric: str = 'euclidean'):
        """
        TopicClustererを初期化
        
        Args:
            min_cluster_size (int): クラスタの最小サイズ
            min_samples (int): クラスタ形成に必要な最小サンプル数
            metric (str): 距離指標 ('euclidean', 'cosine' など)
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan is not installed. Install with: pip install hdbscan")
        
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size は 2 以上である必要があります")
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.clusterer = None
        self.embeddings = None
        self.cluster_labels = None
        self.n_clusters = None
        
        print(f"[OK] TopicClusterer initialized: min_cluster_size={min_cluster_size}, metric={metric}")
    
    def fit(self, embeddings: np.ndarray) -> 'TopicClusterer':
        """
        ベクトルに対してHDBSCANクラスタリングを実行
        
        Args:
            embeddings (np.ndarray): 形状 (N, D) のベクトル配列
            
        Returns:
            TopicClusterer: self
        """
        if embeddings.shape[0] < self.min_cluster_size:
            raise ValueError(f"サンプル数 ({embeddings.shape[0]}) がmin_cluster_size ({self.min_cluster_size}) より少ない")
        
        self.embeddings = embeddings
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric
        )
        self.cluster_labels = self.clusterer.fit_predict(embeddings)
        
        # クラスタ数を計算（-1 はノイズポイント）
        self.n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = np.sum(self.cluster_labels == -1)
        
        print(f"[OK] HDBSCAN fitting completed")
        print(f"[OK] Clusters: {self.n_clusters}, Noise points: {n_noise}")
        
        return self
    
    def fit_predict(self, vectors):
        """後方互換性のためのメソッド"""
        return self.fit(vectors).get_cluster_labels()
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        新しいベクトルをクラスタに割り当てる
        
        Args:
            embeddings (np.ndarray): 新しいベクトル配列
            
        Returns:
            np.ndarray: クラスタラベル
        """
        if self.clusterer is None:
            raise ValueError("fit() を先に実行してください")
        
        # HDBSCANの approximate_predict
        return self.clusterer.approximate_predict(embeddings)[0]
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        クラスタの代表的なベクトルを取得
        
        Returns:
            np.ndarray: クラスタごとの代表ベクトル
        """
        if self.cluster_labels is None:
            raise ValueError("fit() を先に実行してください")
        
        centers = []
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_points = self.embeddings[mask]
            # クラスタ内の平均ベクトルを中心とする
            center = np.mean(cluster_points, axis=0)
            centers.append(center)
        
        return np.array(centers)
    
    def get_cluster_stats(self) -> Dict[int, Dict]:
        """
        各クラスタの統計情報を取得
        
        Returns:
            Dict: クラスタごとの統計情報（サイズ、内部距離等）
        """
        if self.cluster_labels is None:
            raise ValueError("fit() を先に実行してください")
        
        stats = {}
        
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_points = self.embeddings[mask]
            
            # クラスタ内の距離を計算
            center = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - center, axis=1)
            
            stats[cluster_id] = {
                'size': int(np.sum(mask)),
                'mean_distance': float(np.mean(distances)),
                'max_distance': float(np.max(distances)),
                'std_distance': float(np.std(distances))
            }
        
        # ノイズポイントの統計も追加
        noise_mask = self.cluster_labels == -1
        if np.sum(noise_mask) > 0:
            stats['noise'] = {
                'size': int(np.sum(noise_mask)),
                'description': 'Noise points not assigned to any cluster'
            }
        
        return stats
    
    def get_silhouette_score(self) -> Optional[float]:
        """
        シルエット係数を計算
        
        Returns:
            float: シルエット係数 (-1 ～ 1)、またはNoneノイズが多い場合
        """
        if self.embeddings is None or self.cluster_labels is None:
            raise ValueError("fit() を先に実行してください")
        
        # ノイズポイントを除く
        valid_mask = self.cluster_labels != -1
        if np.sum(valid_mask) < 2:
            return None
        
        valid_embeddings = self.embeddings[valid_mask]
        valid_labels = self.cluster_labels[valid_mask]
        
        try:
            return float(silhouette_score(valid_embeddings, valid_labels))
        except Exception:
            return None
    
    def get_davies_bouldin_score(self) -> Optional[float]:
        """
        Davies-Bouldin Index を計算（小さいほど良い）
        
        Returns:
            float: Davies-Bouldin Index、またはNone
        """
        if self.embeddings is None or self.cluster_labels is None:
            raise ValueError("fit() を先に実行してください")
        
        # ノイズポイントを除く
        valid_mask = self.cluster_labels != -1
        if np.sum(valid_mask) < 2:
            return None
        
        valid_embeddings = self.embeddings[valid_mask]
        valid_labels = self.cluster_labels[valid_mask]
        
        try:
            return float(davies_bouldin_score(valid_embeddings, valid_labels))
        except Exception:
            return None
    
    def get_calinski_harabasz_score(self) -> Optional[float]:
        """
        Calinski-Harabasz Index を計算（大きいほど良い）
        
        Returns:
            float: Calinski-Harabasz Index、またはNone
        """
        if self.embeddings is None or self.cluster_labels is None:
            raise ValueError("fit() を先に実行してください")
        
        # ノイズポイントを除く
        valid_mask = self.cluster_labels != -1
        if np.sum(valid_mask) < 2:
            return None
        
        valid_embeddings = self.embeddings[valid_mask]
        valid_labels = self.cluster_labels[valid_mask]
        
        try:
            return float(calinski_harabasz_score(valid_embeddings, valid_labels))
        except Exception:
            return None
    
    def get_cluster_labels(self) -> np.ndarray:
        """
        クラスタラベルを取得（-1はノイズ）
        
        Returns:
            np.ndarray: クラスタラベル配列
        """
        if self.cluster_labels is None:
            raise ValueError("fit() を先に実行してください")
        
        return self.cluster_labels.copy()
    
    def to_dataframe(self, texts: List[str], embeddings: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        テキストとクラスタラベルを DataFrame に変換
        
        Args:
            texts (List[str]): テキストリスト
            embeddings (Optional[np.ndarray]): ベクトル（オプション）
            
        Returns:
            pd.DataFrame: text, cluster, (embedding_*) を含む DataFrame
        """
        if self.cluster_labels is None:
            raise ValueError("fit() を先に実行してください")
        
        if len(texts) != len(self.cluster_labels):
            raise ValueError("テキストとクラスタラベルの数が一致しません")
        
        data = {
            'text': texts,
            'cluster': self.cluster_labels,
            'is_noise': self.cluster_labels == -1
        }
        
        # ベクトル情報を追加（オプション）
        if embeddings is not None:
            for i in range(embeddings.shape[1]):
                data[f'embedding_{i}'] = embeddings[:, i]
        
        df = pd.DataFrame(data)
        return df
    
    def get_cluster_summary(self, texts: List[str], top_n: int = 5) -> Dict[int, Dict]:
        """
        各クラスタの要約情報を取得（代表的なテキスト含む）
        
        Args:
            texts (List[str]): テキストリスト
            top_n (int): 代表テキスト数
            
        Returns:
            Dict: クラスタごとの要約情報
        """
        if self.cluster_labels is None:
            raise ValueError("fit() を先に実行してください")
        
        if len(texts) != len(self.cluster_labels):
            raise ValueError("テキストとクラスタラベルの数が一致しません")
        
        summary = {}
        
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(mask)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_points = self.embeddings[mask]
            
            # クラスタの中心までの距離を計算
            center = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - center, axis=1)
            
            # 中心に最も近いテキストを取得
            top_indices = np.argsort(distances)[:min(top_n, len(cluster_texts))]
            representative_texts = [cluster_texts[i] for i in top_indices]
            
            summary[cluster_id] = {
                'size': len(cluster_texts),
                'representative_texts': representative_texts,
                'stats': self.get_cluster_stats()[cluster_id]
            }
        
        return summary
    
    def __repr__(self) -> str:
        if self.clusterer is None:
            return f"TopicClusterer(min_cluster_size={self.min_cluster_size}, fitted=False)"
        
        return f"TopicClusterer(clusters={self.n_clusters}, fitted=True)"