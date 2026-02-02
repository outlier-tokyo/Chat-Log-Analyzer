"""
CooccurrenceNetworkクラス

単語の共起関係を分析し、ネットワークグラフを構築する
- 共起マトリックスの計算
- ネットワークグラフの構築
- 中心性指標の計算

セキュリティ: ローカルのみ処理
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter
import warnings


class CooccurrenceNetwork:
    """単語共起ネットワーク分析"""
    
    def __init__(self, window_size: int = 5, min_frequency: int = 2):
        """
        CooccurrenceNetworkを初期化
        
        Args:
            window_size (int): 共起の判定ウィンドウサイズ
            min_frequency (int): 最小出現頻度フィルタ
        """
        self.window_size = window_size
        self.min_frequency = min_frequency
        self.cooccurrence_matrix = None
        self.vocabulary = None
        self.graph = None
        self.edge_weights = None
        
        print(f"[OK] CooccurrenceNetwork initialized: window_size={window_size}, min_freq={min_frequency}")
    
    def build_network(self, tokenized_docs: List[List[str]]) -> nx.Graph:
        """
        トークン化されたドキュメントからネットワークを構築
        
        Args:
            tokenized_docs (List[List[str]]): トークン化されたドキュメント
            
        Returns:
            nx.Graph: ネットワークグラフ
        """
        # 共起行列を計算
        self._build_cooccurrence_matrix(tokenized_docs)
        
        # グラフを構築
        self.graph = nx.Graph()
        self.edge_weights = {}
        
        # エッジを追加
        for (word1, word2), weight in self.cooccurrence_matrix.items():
            if weight > 0:
                self.graph.add_edge(word1, word2, weight=weight)
                self.edge_weights[(word1, word2)] = weight
        
        print(f"[OK] Network built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _build_cooccurrence_matrix(self, tokenized_docs: List[List[str]]) -> None:
        """
        共起マトリックスを構築
        
        Args:
            tokenized_docs (List[List[str]]): トークン化されたドキュメント
        """
        cooccurrence = defaultdict(int)
        vocabulary = set()
        
        # 各ドキュメントに対して共起を計算
        for doc in tokenized_docs:
            if len(doc) < 2:
                continue
            
            vocabulary.update(doc)
            
            # ウィンドウ内での共起をカウント
            for i, word1 in enumerate(doc):
                for j in range(max(0, i - self.window_size), min(len(doc), i + self.window_size + 1)):
                    if i != j:
                        word2 = doc[j]
                        # 単語ペアを正規化（順序を固定）
                        key = tuple(sorted([word1, word2]))
                        cooccurrence[key] += 1
        
        # 最小頻度フィルタを適用
        filtered_cooccurrence = {
            k: v for k, v in cooccurrence.items()
            if v >= self.min_frequency
        }
        
        self.cooccurrence_matrix = filtered_cooccurrence
        self.vocabulary = vocabulary
        
        print(f"[OK] Cooccurrence matrix built: {len(vocabulary)} unique words, {len(filtered_cooccurrence)} edges")
    
    def get_cooccurrence_dataframe(self) -> pd.DataFrame:
        """
        共起マトリックスを DataFrame に変換
        
        Returns:
            pd.DataFrame: word1, word2, frequency を含む DataFrame
        """
        if self.cooccurrence_matrix is None:
            raise ValueError("build_network() を先に実行してください")
        
        rows = []
        for (word1, word2), frequency in self.cooccurrence_matrix.items():
            rows.append({'word1': word1, 'word2': word2, 'frequency': frequency})
        
        df = pd.DataFrame(rows)
        return df.sort_values('frequency', ascending=False)
    
    def get_node_degree(self) -> Dict[str, int]:
        """
        各ノードの次数を取得
        
        Returns:
            Dict: 単語ごとの次数
        """
        if self.graph is None:
            raise ValueError("build_network() を先に実行してください")
        
        return dict(self.graph.degree())
    
    def get_node_strength(self) -> Dict[str, float]:
        """
        各ノードの強度（重み付き次数）を取得
        
        Returns:
            Dict: 単語ごとの強度
        """
        if self.graph is None:
            raise ValueError("build_network() を先に実行してください")
        
        strength = {}
        for node in self.graph.nodes():
            total_weight = sum(self.graph[node][neighbor]['weight'] 
                             for neighbor in self.graph.neighbors(node))
            strength[node] = total_weight
        
        return strength
    
    def get_betweenness_centrality(self) -> Dict[str, float]:
        """
        媒介中心性を計算
        
        Returns:
            Dict: 単語ごとの媒介中心性
        """
        if self.graph is None:
            raise ValueError("build_network() を先に実行してください")
        
        if len(self.graph.nodes()) == 0:
            return {}
        
        return nx.betweenness_centrality(self.graph, weight='weight')
    
    def get_closeness_centrality(self) -> Dict[str, float]:
        """
        近接中心性を計算
        
        Returns:
            Dict: 単語ごとの近接中心性
        """
        if self.graph is None:
            raise ValueError("build_network() を先に実行してください")
        
        if len(self.graph.nodes()) == 0:
            return {}
        
        # 連結成分ごとに計算
        result = {}
        for component in nx.connected_components(self.graph):
            subgraph = self.graph.subgraph(component)
            centrality = nx.closeness_centrality(subgraph)
            result.update(centrality)
        
        return result
    
    def get_eigenvector_centrality(self, max_iter: int = 1000) -> Dict[str, float]:
        """
        固有ベクトル中心性を計算
        
        Args:
            max_iter (int): 最大反復回数
            
        Returns:
            Dict: 単語ごとの固有ベクトル中心性
        """
        if self.graph is None:
            raise ValueError("build_network() を先に実行してください")
        
        if len(self.graph.nodes()) == 0:
            return {}
        
        try:
            return nx.eigenvector_centrality(self.graph, weight='weight', max_iter=max_iter)
        except nx.NetworkXError:
            # 収束しない場合は全て 1/n を返す
            n = len(self.graph.nodes())
            return {node: 1.0/n for node in self.graph.nodes()}
    
    def find_communities(self, method: str = 'louvain') -> Dict[int, Set[str]]:
        """
        コミュニティを検出
        
        Args:
            method (str): 検出方法 ('louvain' または 'greedy')
            
        Returns:
            Dict: コミュニティID -> 単語セット
        """
        if self.graph is None:
            raise ValueError("build_network() を先に実行してください")
        
        if len(self.graph.nodes()) == 0:
            return {}
        
        try:
            if method == 'louvain':
                from networkx.algorithms import community
                communities = community.greedy_modularity_communities(self.graph, weight='weight')
            else:
                from networkx.algorithms import community
                communities = community.greedy_modularity_communities(self.graph, weight='weight')
        except Exception as e:
            warnings.warn(f"コミュニティ検出失敗: {e}")
            return {}
        
        result = {}
        for i, community_set in enumerate(communities):
            result[i] = set(community_set)
        
        return result
    
    def get_top_edges(self, top_n: int = 10) -> pd.DataFrame:
        """
        最も重み付けが高いエッジを取得
        
        Args:
            top_n (int): 取得するエッジ数
            
        Returns:
            pd.DataFrame: word1, word2, weight のDataFrame
        """
        if self.cooccurrence_matrix is None:
            raise ValueError("build_network() を先に実行してください")
        
        df = self.get_cooccurrence_dataframe()
        return df.head(top_n)
    
    def get_keywords_by_frequency(self, top_n: int = 20) -> pd.DataFrame:
        """
        頻度が高いキーワードを取得
        
        Args:
            top_n (int): 取得するキーワード数
            
        Returns:
            pd.DataFrame: word, frequency のDataFrame
        """
        if self.cooccurrence_matrix is None:
            raise ValueError("build_network() を先に実行してください")
        
        word_freq = defaultdict(int)
        for (word1, word2), freq in self.cooccurrence_matrix.items():
            word_freq[word1] += freq
            word_freq[word2] += freq
        
        df = pd.DataFrame([
            {'word': word, 'frequency': freq}
            for word, freq in word_freq.items()
        ])
        
        return df.sort_values('frequency', ascending=False).head(top_n)
    
    def __repr__(self) -> str:
        if self.graph is None:
            return f"CooccurrenceNetwork(window_size={self.window_size}, built=False)"
        
        return f"CooccurrenceNetwork(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"