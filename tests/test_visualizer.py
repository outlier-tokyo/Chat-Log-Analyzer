"""
EmbeddingVisualizer テストスイート

テスト対象:
- UMAP 次元圧縮
- インタラクティブな散布図生成
- クラスタ可視化
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import tempfile
import os

# パスを追加してインポート
sys.path.insert(0, str(Path(__file__).parent.parent / 'ai-chat-analyzer' / 'src'))

from visualization.visualizer import EmbeddingVisualizer


class TestEmbeddingVisualizerBasic(unittest.TestCase):
    """基本的な可視化テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        np.random.seed(42)
        cls.embeddings = np.random.randn(100, 50)  # 50次元ベクトル
        cls.visualizer = EmbeddingVisualizer(n_neighbors=15, min_dist=0.1, metric='euclidean')
    
    def test_initialization(self):
        """初期化テスト"""
        self.assertEqual(self.visualizer.n_neighbors, 15)
        self.assertEqual(self.visualizer.min_dist, 0.1)
        self.assertIsNone(self.visualizer.embeddings_2d)
        print("[PASS] 初期化テスト")
    
    def test_fit_transform(self):
        """UMAP fit_transform"""
        embeddings_2d = self.visualizer.fit_transform(self.embeddings)
        
        self.assertEqual(embeddings_2d.shape, (100, 2))
        self.assertIsNotNone(self.visualizer.umap_reducer)
        print("[PASS] UMAP fit_transform テスト")
    
    def test_embeddings_2d_shape(self):
        """2次元埋め込みの形状"""
        embeddings_2d = self.visualizer.embeddings_2d
        
        self.assertEqual(embeddings_2d.shape[0], 100)
        self.assertEqual(embeddings_2d.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(embeddings_2d)))
        print("[PASS] 2次元埋め込み形状テスト")


class TestScatterPlotGeneration(unittest.TestCase):
    """散布図生成テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        np.random.seed(42)
        cls.embeddings = np.random.randn(50, 30)
        cls.visualizer = EmbeddingVisualizer()
        cls.embeddings_2d = cls.visualizer.fit_transform(cls.embeddings)
        cls.texts = [f"Sample {i}" for i in range(50)]
        cls.labels = np.random.randint(0, 3, 50)
    
    def test_basic_scatter_plot(self):
        """基本的な散布図"""
        fig = self.visualizer.plot_scatter(
            embeddings_2d=self.embeddings_2d,
            texts=self.texts,
            title="Test Plot"
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.data[0].x.shape[0], 50)
        print("[PASS] 基本散布図生成テスト")
    
    def test_scatter_with_labels(self):
        """クラスタラベル付き散布図"""
        fig = self.visualizer.plot_scatter(
            embeddings_2d=self.embeddings_2d,
            texts=self.texts,
            labels=self.labels,
            title="Cluster Plot"
        )
        
        self.assertIsNotNone(fig)
        # クラスタがある分、トレースが複数
        self.assertGreater(len(fig.data), 1)
        print(f"[PASS] クラスタ散布図生成テスト（{len(fig.data)}クラスタ）")
    
    def test_cluster_scatter_plot(self):
        """クラスタ別散布図"""
        fig = self.visualizer.plot_cluster_scatter(
            embeddings_2d=self.embeddings_2d,
            texts=self.texts,
            labels=self.labels,
            title="Cluster Scatter"
        )
        
        self.assertIsNotNone(fig)
        print("[PASS] クラスタ別散布図テスト")


class TestHTMLExport(unittest.TestCase):
    """HTML エクスポートテスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        np.random.seed(42)
        cls.embeddings = np.random.randn(30, 20)
        cls.visualizer = EmbeddingVisualizer()
        cls.embeddings_2d = cls.visualizer.fit_transform(cls.embeddings)
        cls.texts = [f"Text {i}" for i in range(30)]
    
    def test_save_html(self):
        """HTML保存"""
        fig = self.visualizer.plot_scatter(
            embeddings_2d=self.embeddings_2d,
            texts=self.texts
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_plot.html")
            self.visualizer.save_html(fig, filepath)
            
            # ファイルが作成されたか確認
            self.assertTrue(os.path.exists(filepath))
            self.assertGreater(os.path.getsize(filepath), 0)
            
            # HTMLコンテンツを確認
            with open(filepath, 'r') as f:
                content = f.read()
                self.assertIn('plotly', content.lower())
            
            print(f"[PASS] HTML保存テスト（{os.path.getsize(filepath)} bytes）")


class TestVisualizerMetrics(unittest.TestCase):
    """可視化指標テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        np.random.seed(42)
        # クラスタ状のデータを生成
        cluster1 = np.random.randn(30, 20)
        cluster2 = np.random.randn(30, 20) + np.array([5] * 20)
        cls.embeddings = np.vstack([cluster1, cluster2])
        cls.visualizer = EmbeddingVisualizer(n_neighbors=10)
        cls.embeddings_2d = cls.visualizer.fit_transform(cls.embeddings)
    
    def test_2d_variance(self):
        """2次元投影の分散"""
        variance = np.var(self.embeddings_2d, axis=0)
        
        # 両軸で分散があることを確認
        self.assertGreater(variance[0], 0)
        self.assertGreater(variance[1], 0)
        print(f"[PASS] 2D分散: ({variance[0]:.3f}, {variance[1]:.3f})")
    
    def test_2d_range(self):
        """2次元投影の範囲"""
        x_range = np.max(self.embeddings_2d[:, 0]) - np.min(self.embeddings_2d[:, 0])
        y_range = np.max(self.embeddings_2d[:, 1]) - np.min(self.embeddings_2d[:, 1])
        
        # 両軸でスケールがあること
        self.assertGreater(x_range, 0)
        self.assertGreater(y_range, 0)
        print(f"[PASS] 2D範囲: X={x_range:.3f}, Y={y_range:.3f}")


class TestEdgeCases(unittest.TestCase):
    """エッジケーステスト"""
    
    def test_small_dataset(self):
        """小さいデータセット"""
        embeddings = np.random.randn(10, 5)
        visualizer = EmbeddingVisualizer(n_neighbors=5)
        
        embeddings_2d = visualizer.fit_transform(embeddings)
        self.assertEqual(embeddings_2d.shape, (10, 2))
        print("[PASS] 小規模データセットテスト")
    
    def test_high_dimensional(self):
        """高次元ベクトル"""
        embeddings = np.random.randn(50, 500)  # 500次元
        visualizer = EmbeddingVisualizer()
        
        embeddings_2d = visualizer.fit_transform(embeddings)
        self.assertEqual(embeddings_2d.shape, (50, 2))
        print("[PASS] 高次元ベクトルテスト")
    
    def test_noise_labels(self):
        """ノイズラベル（-1）を含む"""
        embeddings = np.random.randn(30, 10)
        texts = [f"Text {i}" for i in range(30)]
        labels = np.concatenate([np.array([0, 1] * 14), np.array([-1])])
        
        visualizer = EmbeddingVisualizer()
        embeddings_2d = visualizer.fit_transform(embeddings)
        
        fig = visualizer.plot_scatter(
            embeddings_2d=embeddings_2d,
            texts=texts,
            labels=labels
        )
        
        self.assertIsNotNone(fig)
        print("[PASS] ノイズラベルテスト")


def run_tests():
    """すべてのテストを実行"""
    print("\n" + "=" * 70)
    print("EmbeddingVisualizer テストスイート開始")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingVisualizerBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestScatterPlotGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestHTMLExport))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizerMetrics))
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
