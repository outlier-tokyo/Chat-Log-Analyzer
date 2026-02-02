"""
CooccurrenceNetwork テストスイート

テスト対象:
- 共起ネットワーク構築
- 共起マトリックス計算
- 中心性指標
- ネットワーク統計
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
import sys
from pathlib import Path

# パスを追加してインポート
sys.path.insert(0, str(Path(__file__).parent.parent / 'ai-chat-analyzer' / 'src'))

from analysis.cooccurrence import CooccurrenceNetwork


class TestCooccurrenceNetworkBasic(unittest.TestCase):
    """基本的なネットワーク構築テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        cls.network = CooccurrenceNetwork(window_size=5, min_frequency=2)
        
        # テストデータ：トークン化されたドキュメント
        cls.docs = [
            ['python', 'machine', 'learning', 'neural', 'network'],
            ['python', 'programming', 'language', 'data', 'science'],
            ['machine', 'learning', 'deep', 'learning', 'model'],
            ['neural', 'network', 'training', 'optimization', 'algorithm'],
            ['data', 'science', 'analysis', 'statistics', 'inference']
        ]
    
    def test_initialization(self):
        """初期化テスト"""
        self.assertEqual(self.network.window_size, 5)
        self.assertEqual(self.network.min_frequency, 2)
        self.assertIsNone(self.network.graph)
        print("[PASS] 初期化テスト")
    
    def test_build_network(self):
        """ネットワーク構築"""
        graph = self.network.build_network(self.docs)
        
        self.assertIsNotNone(graph)
        self.assertGreater(graph.number_of_nodes(), 0)
        self.assertGreater(graph.number_of_edges(), 0)
        print(f"[PASS] ネットワーク構築: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    def test_network_properties(self):
        """ネットワークプロパティ"""
        self.assertIsNotNone(self.network.graph)
        self.assertTrue(isinstance(self.network.graph, nx.Graph))
        self.assertTrue(self.network.graph.number_of_nodes() > 0)
        print("[PASS] ネットワークプロパティテスト")


class TestCooccurrenceMatrixConversion(unittest.TestCase):
    """共起マトリックス変換テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        cls.network = CooccurrenceNetwork(window_size=3, min_frequency=1)
        
        cls.docs = [
            ['apple', 'banana', 'orange', 'fruit'],
            ['banana', 'orange', 'tropical', 'fruit'],
            ['apple', 'fruit', 'food', 'healthy']
        ]
        
        cls.network.build_network(cls.docs)
    
    def test_cooccurrence_dataframe(self):
        """共起マトリックスのDataFrame変換"""
        df = self.network.get_cooccurrence_dataframe()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('word1', df.columns)
        self.assertIn('word2', df.columns)
        self.assertIn('frequency', df.columns)
        self.assertTrue(len(df) > 0)
        print(f"[PASS] 共起DataFrame: {len(df)} エッジ")
    
    def test_frequency_order(self):
        """周波数の降順"""
        df = self.network.get_cooccurrence_dataframe()
        
        # 周波数が降順になっていることを確認
        freqs = df['frequency'].values
        self.assertEqual(list(freqs), sorted(freqs, reverse=True))
        print("[PASS] 周波数降順確認")


class TestCentralityMeasures(unittest.TestCase):
    """中心性指標テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        cls.network = CooccurrenceNetwork(window_size=4, min_frequency=1)
        
        cls.docs = [
            ['data', 'science', 'machine', 'learning'],
            ['machine', 'learning', 'artificial', 'intelligence'],
            ['data', 'analysis', 'statistics', 'inference'],
            ['neural', 'network', 'deep', 'learning'],
            ['learning', 'algorithm', 'optimization', 'model']
        ]
        
        cls.network.build_network(cls.docs)
    
    def test_node_degree(self):
        """ノードの次数"""
        degree = self.network.get_node_degree()
        
        self.assertIsInstance(degree, dict)
        self.assertGreater(len(degree), 0)
        self.assertTrue(all(v > 0 for v in degree.values()))
        print(f"[PASS] ノード次数: {len(degree)}ノード")
    
    def test_node_strength(self):
        """ノードの強度（重み付き次数）"""
        strength = self.network.get_node_strength()
        
        self.assertIsInstance(strength, dict)
        self.assertGreater(len(strength), 0)
        self.assertTrue(all(v > 0 for v in strength.values()))
        print(f"[PASS] ノード強度: 平均 {np.mean(list(strength.values())):.2f}")
    
    def test_betweenness_centrality(self):
        """媒介中心性"""
        centrality = self.network.get_betweenness_centrality()
        
        self.assertIsInstance(centrality, dict)
        # 空でない場合、値は0以上1以下
        if centrality:
            self.assertTrue(all(0 <= v <= 1 for v in centrality.values()))
        print(f"[PASS] 媒介中心性計算完了")
    
    def test_closeness_centrality(self):
        """近接中心性"""
        centrality = self.network.get_closeness_centrality()
        
        self.assertIsInstance(centrality, dict)
        if centrality:
            self.assertTrue(all(0 <= v <= 1 for v in centrality.values()))
        print(f"[PASS] 近接中心性計算完了")
    
    def test_eigenvector_centrality(self):
        """固有ベクトル中心性"""
        centrality = self.network.get_eigenvector_centrality()
        
        self.assertIsInstance(centrality, dict)
        if centrality:
            self.assertTrue(all(v >= 0 for v in centrality.values()))
        print(f"[PASS] 固有ベクトル中心性計算完了")


class TestKeywordExtraction(unittest.TestCase):
    """キーワード抽出テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        cls.network = CooccurrenceNetwork(window_size=3, min_frequency=1)
        
        cls.docs = [
            ['python', 'programming', 'code'],
            ['python', 'data', 'analysis'],
            ['python', 'machine', 'learning'],
            ['programming', 'code', 'development'],
            ['data', 'analysis', 'statistics']
        ]
        
        cls.network.build_network(cls.docs)
    
    def test_top_edges(self):
        """最も重いエッジ"""
        top_edges = self.network.get_top_edges(top_n=5)
        
        self.assertIsInstance(top_edges, pd.DataFrame)
        self.assertLessEqual(len(top_edges), 5)
        if len(top_edges) > 1:
            # 周波数が降順になっている
            self.assertTrue(
                all(top_edges['frequency'].iloc[i] >= top_edges['frequency'].iloc[i+1]
                    for i in range(len(top_edges)-1))
            )
        print(f"[PASS] トップエッジ: {len(top_edges)}個")
    
    def test_keywords_by_frequency(self):
        """頻度が高いキーワード"""
        keywords = self.network.get_keywords_by_frequency(top_n=10)
        
        self.assertIsInstance(keywords, pd.DataFrame)
        self.assertIn('word', keywords.columns)
        self.assertIn('frequency', keywords.columns)
        self.assertLessEqual(len(keywords), 10)
        print(f"[PASS] トップキーワード: {len(keywords)}個")


class TestEdgeCases(unittest.TestCase):
    """エッジケーステスト"""
    
    def test_empty_documents(self):
        """空のドキュメント"""
        network = CooccurrenceNetwork(window_size=2, min_frequency=1)
        
        # 空ドキュメントでもエラーにならない
        graph = network.build_network([[], []])
        self.assertIsNotNone(graph)
        print("[PASS] 空ドキュメント処理")
    
    def test_single_word_documents(self):
        """単一単語ドキュメント"""
        network = CooccurrenceNetwork(window_size=2, min_frequency=1)
        
        docs = [['word'], ['word'], ['word']]
        graph = network.build_network(docs)
        
        # 共起がないためエッジは0
        self.assertEqual(graph.number_of_edges(), 0)
        print("[PASS] 単一単語ドキュメント処理")
    
    def test_min_frequency_filter(self):
        """最小周波数フィルタ"""
        network = CooccurrenceNetwork(window_size=2, min_frequency=5)
        
        docs = [
            ['a', 'b'], ['a', 'b'],  # a-b: 2回
            ['c', 'd'], ['c', 'd'], ['c', 'd'], ['c', 'd'], ['c', 'd']  # c-d: 5回
        ]
        graph = network.build_network(docs)
        
        # min_frequency=5なので、c-dだけが残る
        self.assertLessEqual(graph.number_of_edges(), 1)
        print(f"[PASS] 最小周波数フィルタ: {graph.number_of_edges()}エッジ")


def run_tests():
    """すべてのテストを実行"""
    print("\n" + "=" * 70)
    print("CooccurrenceNetwork テストスイート開始")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCooccurrenceNetworkBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestCooccurrenceMatrixConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestCentralityMeasures))
    suite.addTests(loader.loadTestsFromTestCase(TestKeywordExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print(f"テスト完了: {result.testsRun}件実行")
    print(f"[OK] 成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"[NG] 失敗: {len(result.failures)}")
    print(f"[NG] エラー: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
