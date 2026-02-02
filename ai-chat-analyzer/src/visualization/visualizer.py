"""
可視化クラス

UMAP + Plotly を使用したインタラクティブな散布図表示
- 2次元UMAP投影
- クラスタ色分け
- ホバー情報表示
- 対話的なズーム・パン
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional, Dict
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("umap-learn not installed. Install with: pip install umap-learn")


class EmbeddingVisualizer:
    """ベクトル埋め込み可視化クラス"""
    
    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'euclidean'):
        """
        EmbeddingVisualizerを初期化
        
        Args:
            n_neighbors (int): UMAP の近傍数
            min_dist (float): UMAP の最小距離
            metric (str): 距離指標 ('euclidean', 'cosine' など)
        """
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is not installed. Install with: pip install umap-learn")
        
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.umap_reducer = None
        self.embeddings_2d = None
        
        print(f"[OK] EmbeddingVisualizer initialized: n_neighbors={n_neighbors}, metric={metric}")
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        ベクトルを2次元に投影
        
        Args:
            embeddings (np.ndarray): 形状 (N, D) のベクトル配列
            
        Returns:
            np.ndarray: 形状 (N, 2) の2次元投影
        """
        try:
            # Try UMAP
            self.umap_reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(self.n_neighbors, len(embeddings) - 1),
                min_dist=self.min_dist,
                metric=self.metric,
                verbose=0
            )
            
            self.embeddings_2d = self.umap_reducer.fit_transform(embeddings)
            print(f"[OK] UMAP fit_transform completed")
            
        except Exception as e:
            # Fallback to PCA if UMAP fails
            print(f"[WARN] UMAP failed ({type(e).__name__}), falling back to PCA")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            self.embeddings_2d = pca.fit_transform(embeddings)
            print(f"[OK] PCA fit_transform completed")
        
        print(f"[OK] Embeddings shape: {self.embeddings_2d.shape}")
        
        return self.embeddings_2d
    
    def plot_scatter(
        self,
        embeddings_2d: Optional[np.ndarray] = None,
        texts: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
        title: str = "Embedding Visualization",
        height: int = 700,
        width: int = 1000,
        show_grid: bool = True
    ) -> go.Figure:
        """
        インタラクティブな散布図を作成
        
        Args:
            embeddings_2d (Optional[np.ndarray]): 2次元ベクトル (N, 2)
            texts (Optional[List[str]]): ホバー表示用のテキスト
            labels (Optional[np.ndarray]): クラスタラベル（色分け用）
            title (str): グラフタイトル
            height (int): グラフ高さ
            width (int): グラフ幅
            show_grid (bool): グリッド表示
            
        Returns:
            go.Figure: Plotly フィギュア
        """
        if embeddings_2d is None:
            if self.embeddings_2d is None:
                raise ValueError("fit_transform() を先に実行してください")
            embeddings_2d = self.embeddings_2d
        
        # DataFrame を作成
        data = {
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1]
        }
        
        if texts is not None:
            data['text'] = texts
        else:
            data['text'] = [f"Point {i}" for i in range(len(embeddings_2d))]
        
        if labels is not None:
            data['cluster'] = labels.astype(str)
        else:
            data['cluster'] = '0'
        
        df = pd.DataFrame(data)
        
        # Plotly Express で散布図を作成
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data={'text': True, 'x': ':.3f', 'y': ':.3f'},
            title=title,
            labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
            height=height,
            width=width,
            hover_name='text'
        )
        
        # グリッド設定
        fig.update_xaxes(showgrid=show_grid)
        fig.update_yaxes(showgrid=show_grid)
        
        # マーカー設定
        fig.update_traces(
            marker=dict(
                size=8,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            )
        )
        
        # レイアウト設定
        fig.update_layout(
            hovermode='closest',
            plot_bgcolor='#f8f9fa',
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        print(f"[OK] Scatter plot created: {len(df)} points")
        
        return fig
    
    def plot_cluster_scatter(
        self,
        embeddings_2d: Optional[np.ndarray] = None,
        texts: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
        title: str = "Cluster Visualization",
        height: int = 700,
        width: int = 1000
    ) -> go.Figure:
        """
        クラスタ別に色分けした散布図
        
        Args:
            embeddings_2d (Optional[np.ndarray]): 2次元ベクトル
            texts (Optional[List[str]]): ホバー用テキスト
            labels (Optional[np.ndarray]): クラスタラベル
            title (str): タイトル
            height (int): 高さ
            width (int): 幅
            
        Returns:
            go.Figure: Plotly フィギュア
        """
        if embeddings_2d is None:
            if self.embeddings_2d is None:
                raise ValueError("fit_transform() を先に実行してください")
            embeddings_2d = self.embeddings_2d
        
        fig = self.plot_scatter(
            embeddings_2d=embeddings_2d,
            texts=texts,
            labels=labels,
            title=title,
            height=height,
            width=width,
            show_grid=True
        )
        
        # クラスタ色を統一
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = px.colors.qualitative.Set1
            if len(unique_labels) <= 10:
                colors = px.colors.qualitative.Set1
            elif len(unique_labels) <= 20:
                colors = px.colors.qualitative.Light24
            else:
                colors = px.colors.sample_colorscales("Viridis")[0]
            
            fig.for_each_trace(lambda trace: trace.update(
                marker_color=colors[int(trace.name) % len(colors)]
            ))
        
        return fig
    
    def save_html(self, fig: go.Figure, filepath: str) -> None:
        """
        HTML ファイルに保存
        
        Args:
            fig (go.Figure): Plotly フィギュア
            filepath (str): 保存先パス
        """
        fig.write_html(filepath)
        print(f"[OK] Saved to {filepath}")
    
    def show(self, fig: go.Figure) -> None:
        """
        ブラウザで表示
        
        Args:
            fig (go.Figure): Plotly フィギュア
        """
        fig.show()
        print(f"[OK] Opening in browser")
    
    def __repr__(self) -> str:
        return f"EmbeddingVisualizer(n_neighbors={self.n_neighbors}, metric={self.metric})"
