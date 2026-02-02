"""
TopicClusterer テストスイート

テスト対象:
- HDBSCAN クラスタリング
- クラスタ統計情報
- ノイズ点の検出
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# パスを追加してインポート
sys.path.insert(0, str(Path(__file__).parent.parent / 'ai-chat-analyzer' / 'src'))

from analysis.clustering import TopicClusterer


class TestTopicClustererBasic(unittest.TestCase):
    """基本的なクラスタリングテスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        np.random.seed(42)
        # クラスタ状のデータを生成
        cluster1 = np.random.randn(30, 10) + np.array([0] * 10)
        cluster2 = np.random.randn(30, 10) + np.array([5] * 10)
        cluster3 = np.random.randn(30, 10) + np.array([-5] * 10)
        cls.embeddings = np.vstack([cluster1, cluster2, cluster3])
    
    def setUp(self):
        """各テスト前にクラスタラーを初期化"""
        self.clusterer = TopicClusterer(min_cluster_size=10, min_samples=5)
    
    def test_initialization(self):
        """初期化テスト"""
        self.assertEqual(self.clusterer.min_cluster_size, 10)
        self.assertIsNone(self.clusterer.clusterer)
        print("[PASS] 初期化テスト")
    
    def test_fit(self):
        """フィッティングテスト"""
        self.clusterer.fit(self.embeddings)
        
        self.assertIsNotNone(self.clusterer.clusterer)
        self.assertEqual(len(self.clusterer.cluster_labels), 90)
        self.assertGreater(self.clusterer.n_clusters, 0)
        print(f"[PASS] フィッティングテスト（{self.clusterer.n_clusters}クラスタ検出）")
    
    def test_cluster_labels_range(self):
        """クラスタラベルの範囲（-1はノイズ）"""
        self.clusterer.fit(self.embeddings)
        labels = self.clusterer.get_cluster_labels()
        
        self.assertEqual(len(labels), 90)
        self.assertTrue(np.all(labels >= -1))
        print("[PASS] クラスタラベル範囲テスト")
    
    def test_get_cluster_centers(self):
        """クラスタ中心の形状"""
        self.clusterer.fit(self.embeddings)
        centers = self.clusterer.get_cluster_centers()
        
        self.assertEqual(centers.shape[1], 10)
        self.assertEqual(centers.shape[0], self.clusterer.n_clusters)
        print("[PASS] クラスタ中心形状テスト")


class TestTopicClustererStats(unittest.TestCase):
    """統計情報テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        np.random.seed(42)
        cluster1 = np.random.randn(30, 10) + np.array([0] * 10)
        cluster2 = np.random.randn(30, 10) + np.array([5] * 10)
        cluster3 = np.random.randn(30, 10) + np.array([-5] * 10)
        cls.embeddings = np.vstack([cluster1, cluster2, cluster3])
        cls.clusterer = TopicClusterer(min_cluster_size=10, min_samples=5)
        cls.clusterer.fit(cls.embeddings)
    
    def test_cluster_stats(self):
        """クラスタ統計情報"""
        stats = self.clusterer.get_cluster_stats()
        
        # クラスタ数＋ノイズ
        expected_keys = self.clusterer.n_clusters + (1 if any(k == 'noise' for k in stats.keys()) else 0)
        self.assertGreaterEqual(len(stats), self.clusterer.n_clusters)
        
        for cluster_id, stat in stats.items():
            if cluster_id != 'noise':
                self.assertIn('size', stat)
                self.assertIn('mean_distance', stat)
                self.assertGreater(stat['size'], 0)
        
        print("[PASS] クラスタ統計テスト")
    
    def test_silhouette_score(self):
        """シルエット係数"""
        score = self.clusterer.get_silhouette_score()
        
        if score is not None:
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)
            print(f"[PASS] シルエット係数: {score:.3f}")
        else:
            print("[SKIP] シルエット係数: ノイズが多い")
    
    def test_davies_bouldin_score(self):
        """Davies-Bouldin Index"""
        score = self.clusterer.get_davies_bouldin_score()
        
        if score is not None:
            self.assertGreaterEqual(score, 0.0)
            print(f"[PASS] Davies-Bouldin: {score:.3f}")
        else:
            print("[SKIP] Davies-Bouldin: ノイズが多い")
    
    def test_calinski_harabasz_score(self):
        """Calinski-Harabasz Index"""
        score = self.clusterer.get_calinski_harabasz_score()
        
        if score is not None:
            self.assertGreater(score, 0.0)
            print(f"[PASS] Calinski-Harabasz: {score:.3f}")
        else:
            print("[SKIP] Calinski-Harabasz: ノイズが多い")


class TestTopicClustererDataFrame(unittest.TestCase):
    """DataFrame変換テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        np.random.seed(42)
        cluster1 = np.random.randn(25, 10) + np.array([0] * 10)
        cluster2 = np.random.randn(25, 10) + np.array([5] * 10)
        cls.embeddings = np.vstack([cluster1, cluster2])
        cls.texts = [f"テキスト{i}" for i in range(50)]
        cls.clusterer = TopicClusterer(min_cluster_size=10, min_samples=3)
        cls.clusterer.fit(cls.embeddings)
    
    def test_to_dataframe_basic(self):
        """基本的なDataFrame変換"""
        df = self.clusterer.to_dataframe(self.texts)
        
        self.assertEqual(len(df), 50)
        self.assertIn('text', df.columns)
        self.assertIn('cluster', df.columns)
        self.assertIn('is_noise', df.columns)
        print("[PASS] DataFrame変換テスト")
    
    def test_to_dataframe_with_embeddings(self):
        """ベクトル付きDataFrame変換"""
        df = self.clusterer.to_dataframe(self.texts, self.embeddings)
        
        self.assertEqual(len(df), 50)
        self.assertIn('text', df.columns)
        self.assertIn('cluster', df.columns)
        self.assertIn('is_noise', df.columns)
        self.assertIn('embedding_0', df.columns)
        self.assertEqual(df.shape[1], 13)  # text + cluster + is_noise + 10 embeddings
        print("[PASS] ベクトル付きDataFrame変換テスト")


class TestTopicClustererSummary(unittest.TestCase):
    """クラスタ要約テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        np.random.seed(42)
        cluster1 = np.random.randn(40, 10) + np.array([0] * 10)
        cluster2 = np.random.randn(40, 10) + np.array([5] * 10)
        cluster3 = np.random.randn(20, 10) + np.array([-5] * 10)
        cls.embeddings = np.vstack([cluster1, cluster2, cluster3])
        cls.texts = [f"テキスト{i}" for i in range(100)]
        cls.clusterer = TopicClusterer(min_cluster_size=15, min_samples=5)
        cls.clusterer.fit(cls.embeddings)
    
    def test_cluster_summary(self):
        """クラスタ要約情報"""
        summary = self.clusterer.get_cluster_summary(self.texts, top_n=3)
        
        self.assertEqual(len(summary), self.clusterer.n_clusters)
        for cluster_id, info in summary.items():
            self.assertIn('size', info)
            self.assertIn('representative_texts', info)
            self.assertIn('stats', info)
            self.assertLessEqual(len(info['representative_texts']), 3)
        
        print("[PASS] クラスタ要約テスト")


class TestTopicClustererEdgeCases(unittest.TestCase):
    """エッジケーステスト"""
    
    def test_invalid_cluster_size(self):
        """無効なクラスタサイズ"""
        with self.assertRaises(ValueError):
            TopicClusterer(min_cluster_size=1)
    
    def test_insufficient_samples(self):
        """サンプル不足"""
        clusterer = TopicClusterer(min_cluster_size=100)
        embeddings = np.random.randn(50, 10)
        
        with self.assertRaises(ValueError):
            clusterer.fit(embeddings)
        
        print("[PASS] サンプル不足エラーテスト")
    
    def test_predict_without_fit(self):
        """fit前のpredict"""
        clusterer = TopicClusterer(min_cluster_size=10)
        embeddings = np.random.randn(20, 10)
        
        with self.assertRaises(ValueError):
            clusterer.predict(embeddings)
        
        print("[PASS] fit前predict エラーテスト")
    
    def test_noise_detection(self):
        """ノイズ点の検出"""
        np.random.seed(42)
        cluster1 = np.random.randn(30, 10) + np.array([0] * 10)
        cluster2 = np.random.randn(30, 10) + np.array([10] * 10)
        # ノイズポイント
        noise = np.random.uniform(-50, 50, (10, 10))
        embeddings = np.vstack([cluster1, cluster2, noise])
        
        clusterer = TopicClusterer(min_cluster_size=15, min_samples=5)
        clusterer.fit(embeddings)
        
        labels = clusterer.get_cluster_labels()
        noise_count = np.sum(labels == -1)
        
        # ノイズが検出されたことを確認
        self.assertGreater(noise_count, 0)
        print(f"[PASS] ノイズ検出テスト（{noise_count}個のノイズ点検出）")


def run_tests():
    """すべてのテストを実行"""
    print("\n" + "=" * 70)
    print("TopicClusterer テストスイート開始")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestTopicClustererBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestTopicClustererStats))
    suite.addTests(loader.loadTestsFromTestCase(TestTopicClustererDataFrame))
    suite.addTests(loader.loadTestsFromTestCase(TestTopicClustererSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestTopicClustererEdgeCases))
    
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
