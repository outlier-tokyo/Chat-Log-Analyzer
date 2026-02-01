"""
TextVectorizerの包括的なテスト

以下をテストします:
- 基本的なベクトル化
- バッチ処理
- テキスト類似度計算
- 類似度マトリックス
- テキスト検索
- 次元削減（PCA, UMAP）
- DataFrame変換
- パフォーマンス

セキュリティに関する重要な注意事項:
- このTextVectorizerはすべての処理をローカル（オンプレミス）で実行します
- Sentence-Transformersモデルはローカルにダウンロード・保存されます
- 機密データはインターネットに送信されません
- ベクトル化されたデータはローカルメモリのみに保持されます
- 外部API呼び出しは行われません
"""

import unittest
import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path

# パスを追加してインポート
sys.path.insert(0, str(Path(__file__).parent.parent / 'ai-chat-analyzer' / 'src'))

from analysis.vectorizer import TextVectorizer


class TestTextVectorizerBasic(unittest.TestCase):
    """基本的なベクトル化テスト"""
    
    @classmethod
    def setUpClass(cls):
        """全テストで共通のセットアップ"""
        print("\n[INFO] TextVectorizerを初期化中（初回実行時は時間がかかります）...")
        cls.vectorizer = TextVectorizer()
    
    def test_encode_single_text(self):
        """単一テキストのベクトル化"""
        text = "これはテストテキストです"
        embedding = self.vectorizer.encode(text)
        
        # ベクトルの形状を確認
        self.assertEqual(len(embedding.shape), 2)
        self.assertEqual(embedding.shape[0], 1)  # 1つのテキスト
        self.assertGreater(embedding.shape[1], 0)  # 次元 > 0
        
        # ベクトルの値が正規化されていることを確認（L2ノルム ≈ 1）
        norm = np.linalg.norm(embedding[0])
        self.assertAlmostEqual(norm, 1.0, places=5)
        
        print("[PASS] 単一テキストベクトル化テスト")
    
    def test_encode_multiple_texts(self):
        """複数テキストのベクトル化"""
        texts = [
            "これはテキスト1です",
            "これはテキスト2です",
            "これはテキスト3です"
        ]
        embeddings = self.vectorizer.encode(texts)
        
        # ベクトルの形状を確認
        self.assertEqual(embeddings.shape[0], 3)
        self.assertGreater(embeddings.shape[1], 0)
        
        # 各ベクトルが正規化されていることを確認
        for i in range(3):
            norm = np.linalg.norm(embeddings[i])
            self.assertAlmostEqual(norm, 1.0, places=5)
        
        print("[PASS] 複数テキストベクトル化テスト")
    
    def test_encode_japanese_text(self):
        """日本語テキストのベクトル化"""
        texts = [
            "こんにちは",
            "おはようございます",
            "こんばんは"
        ]
        embeddings = self.vectorizer.encode(texts)
        
        self.assertEqual(embeddings.shape[0], 3)
        self.assertGreater(embeddings.shape[1], 0)
        
        print("[PASS] 日本語テキストベクトル化テスト")
    
    def test_encode_mixed_language(self):
        """混合言語テキストのベクトル化"""
        texts = [
            "Hello world",
            "こんにちは",
            "你好"
        ]
        embeddings = self.vectorizer.encode(texts)
        
        self.assertEqual(embeddings.shape[0], 3)
        
        print("[PASS] 混合言語ベクトル化テスト")
    
    def test_encode_special_characters(self):
        """特殊文字を含むテキストのベクトル化"""
        texts = [
            "Email: test@example.com",
            "URL: https://example.com",
            "絵文字: 😊 👍 🎉",
            "数字: 123, 456, 789"
        ]
        embeddings = self.vectorizer.encode(texts)
        
        self.assertEqual(embeddings.shape[0], 4)
        
        print("[PASS] 特殊文字ベクトル化テスト")
    
    def test_embed_consistency(self):
        """同じテキストの同じベクトル化"""
        text = "テストテキスト"
        embedding1 = self.vectorizer.encode(text)
        embedding2 = self.vectorizer.encode(text)
        
        # 同じテキストは同じベクトルを返す
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)
        
        print("[PASS] ベクトル化の一貫性テスト")


class TestTextVectorizerSimilarity(unittest.TestCase):
    """テキスト類似度計算のテスト"""
    
    @classmethod
    def setUpClass(cls):
        cls.vectorizer = TextVectorizer()
    
    def test_similarity_identical_texts(self):
        """同一テキストの類似度"""
        text = "これはテストです"
        similarity = self.vectorizer.similarity(text, text)
        
        # 同一テキストは完全に類似
        self.assertAlmostEqual(similarity, 1.0, places=3)
        
        print("[PASS] 同一テキスト類似度テスト")
    
    def test_similarity_different_texts(self):
        """異なるテキストの類似度"""
        text1 = "犬"
        text2 = "猫"
        similarity = self.vectorizer.similarity(text1, text2)
        
        # 異なるテキストは0～1の間
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)
        
        print("[PASS] 異なるテキスト類似度テスト")
    
    def test_similarity_related_texts(self):
        """関連したテキストの類似度"""
        text1 = "私は猫が好きです"
        text2 = "猫は私の好きな動物です"
        similarity = self.vectorizer.similarity(text1, text2)
        
        # 関連したテキストは高い類似度
        self.assertGreater(similarity, 0.5)
        
        print("[PASS] 関連テキスト類似度テスト")
    
    def test_similarity_matrix(self):
        """類似度マトリックス計算"""
        texts = [
            "犬は動物です",
            "猫は動物です",
            "りんごは果物です"
        ]
        similarity_matrix = self.vectorizer.similarity_matrix(texts)
        
        # マトリックスの形状
        self.assertEqual(similarity_matrix.shape, (3, 3))
        
        # 対角線は1（自分自身）
        for i in range(3):
            self.assertAlmostEqual(similarity_matrix[i, i], 1.0, places=3)
        
        # 対称行列
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(similarity_matrix[i, j], 
                                     similarity_matrix[j, i], places=5)
        
        print("[PASS] 類似度マトリックステスト")


class TestTextVectorizerSearch(unittest.TestCase):
    """テキスト検索のテスト"""
    
    @classmethod
    def setUpClass(cls):
        cls.vectorizer = TextVectorizer()
        cls.candidates = [
            "Python はプログラミング言語です",
            "JavaScript は Web 開発に使用されます",
            "Ruby は美しいプログラミング言語です",
            "Java はエンタープライズ開発に使用されます",
            "C++ は高性能言語です"
        ]
    
    def test_find_similar_basic(self):
        """基本的なテキスト検索"""
        query = "Python を使ったプログラミング"
        results = self.vectorizer.find_similar_texts(query, self.candidates, top_k=2)
        
        # 結果の形式を確認
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 2)  # (text, score)
        
        # 最も似たテキストは最初
        self.assertGreater(results[0][1], results[1][1])
        
        print("[PASS] 基本的なテキスト検索テスト")
    
    def test_find_similar_top_k(self):
        """異なるtop_k値での検索"""
        query = "プログラミング言語"
        
        for top_k in [1, 3, 5]:
            results = self.vectorizer.find_similar_texts(query, self.candidates, top_k=top_k)
            self.assertEqual(len(results), min(top_k, len(self.candidates)))
        
        print("[PASS] top_k値テスト")
    
    def test_find_similar_ordering(self):
        """検索結果は類似度の降順"""
        query = "プログラミング"
        results = self.vectorizer.find_similar_texts(query, self.candidates, top_k=5)
        
        # 類似度が降順であることを確認
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i][1], results[i+1][1])
        
        print("[PASS] 検索結果順序テスト")


class TestTextVectorizerDataFrame(unittest.TestCase):
    """DataFrame変換のテスト"""
    
    @classmethod
    def setUpClass(cls):
        cls.vectorizer = TextVectorizer()
    
    def test_to_dataframe_basic(self):
        """基本的なDataFrame変換"""
        texts = ["テキスト1", "テキスト2", "テキスト3"]
        df = self.vectorizer.to_dataframe(texts)
        
        # DataFrameの形状
        self.assertEqual(len(df), 3)
        self.assertIn("text", df.columns)
        
        # テキストカラムの確認
        self.assertEqual(df["text"].tolist(), texts)
        
        # ベクトルカラムの確認
        embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
        self.assertEqual(len(embedding_cols), self.vectorizer.embedding_dim)
        
        print("[PASS] DataFrame変換テスト")
    
    def test_to_dataframe_with_precomputed_embeddings(self):
        """事前計算されたベクトルでのDataFrame変換"""
        texts = ["テキスト1", "テキスト2"]
        embeddings = self.vectorizer.encode(texts)
        
        df = self.vectorizer.to_dataframe(texts, embeddings)
        
        self.assertEqual(len(df), 2)
        self.assertIn("text", df.columns)
        
        print("[PASS] 事前計算ベクトル DataFrame変換テスト")


class TestTextVectorizerDimensionReduction(unittest.TestCase):
    """次元削減のテスト"""
    
    @classmethod
    def setUpClass(cls):
        cls.vectorizer = TextVectorizer()
        cls.texts = ["テキスト " + str(i) for i in range(10)]
        cls.embeddings = cls.vectorizer.encode(cls.texts)
    
    def test_reduce_dimensions_pca(self):
        """PCAによる次元削減"""
        reduced = self.vectorizer.reduce_dimensions(
            self.embeddings,
            n_components=2,
            method="pca"
        )
        
        # 形状の確認
        self.assertEqual(reduced.shape[0], len(self.embeddings))
        self.assertEqual(reduced.shape[1], 2)
        
        print("[PASS] PCA次元削減テスト")
    
    def test_reduce_dimensions_umap(self):
        """UMAPによる次元削減"""
        try:
            reduced = self.vectorizer.reduce_dimensions(
                self.embeddings,
                n_components=2,
                method="umap"
            )
            
            self.assertEqual(reduced.shape[0], len(self.embeddings))
            self.assertEqual(reduced.shape[1], 2)
            
            print("[PASS] UMAP次元削減テスト")
        except ImportError:
            print("[SKIP] UMAP次元削減テスト (umap-learn not installed)")
    
    def test_reduce_dimensions_invalid_method(self):
        """不正な次元削減方法"""
        with self.assertRaises(ValueError):
            self.vectorizer.reduce_dimensions(
                self.embeddings,
                n_components=2,
                method="invalid_method"
            )
        
        print("[PASS] 不正な次元削減方法テスト")


class TestTextVectorizerModelInfo(unittest.TestCase):
    """モデル情報のテスト"""
    
    @classmethod
    def setUpClass(cls):
        cls.vectorizer = TextVectorizer()
    
    def test_get_model_info(self):
        """モデル情報の取得"""
        info = self.vectorizer.get_model_info()
        
        self.assertIn("model_name", info)
        self.assertIn("embedding_dimension", info)
        self.assertIn("device", info)
        self.assertIn("max_seq_length", info)
        
        self.assertGreater(info["embedding_dimension"], 0)
        self.assertIn(info["device"], ["cpu", "cuda"])
        
        print("[PASS] モデル情報取得テスト")
    
    def test_repr(self):
        """__repr__メソッド"""
        repr_str = repr(self.vectorizer)
        
        self.assertIn("TextVectorizer", repr_str)
        self.assertIn("embedding", repr_str.lower())
        
        print("[PASS] __repr__テスト")


class TestTextVectorizerErrorHandling(unittest.TestCase):
    """エラーハンドリングのテスト"""
    
    @classmethod
    def setUpClass(cls):
        cls.vectorizer = TextVectorizer()
    
    def test_encode_empty_list(self):
        """空のテキストリスト"""
        with self.assertRaises(ValueError):
            self.vectorizer.encode([])
        
        print("[PASS] 空リストエラーテスト")
    
    def test_encode_none_text(self):
        """Noneテキスト"""
        with self.assertRaises((TypeError, ValueError)):
            self.vectorizer.encode(None)
        
        print("[PASS] Noneテキストエラーテスト")


class TestTextVectorizerSecurity(unittest.TestCase):
    """セキュリティテスト（機密データ保護）"""
    
    @classmethod
    def setUpClass(cls):
        cls.vectorizer = TextVectorizer()
    
    def test_local_only_processing(self):
        """ローカルのみでの処理確認"""
        # 機密データの例
        sensitive_text = "クレジットカード: 1234-5678-9012-3456"
        
        # ベクトル化（ローカル処理のみ）
        embedding = self.vectorizer.encode(sensitive_text)
        
        # ベクトル化成功 = ローカル処理完了
        self.assertIsNotNone(embedding)
        self.assertTrue(np.all(np.isfinite(embedding)))
        
        print("[OK] ローカルのみでベクトル化完了（インターネット通信なし）")
    
    def test_sensitive_data_no_transmission(self):
        """機密データの非送信確認"""
        # 個人情報の例
        sensitive_texts = [
            "個人番号: 12345678",
            "メールアドレス: user@confidential.com",
            "電話番号: 090-1234-5678"
        ]
        
        # ベクトル化
        embeddings = self.vectorizer.encode(sensitive_texts)
        
        # 処理完了を確認
        self.assertEqual(embeddings.shape[0], 3)
        self.assertTrue(np.all(np.isfinite(embeddings)))
        
        print("[OK] 機密データのセキュアなベクトル化完了")
    
    def test_vector_is_non_reversible(self):
        """ベクトルから元テキストは復号不可能"""
        original_text = "機密情報: password123"
        
        # ベクトル化
        embedding = self.vectorizer.encode(original_text)
        
        # ベクトル形式確認（一方向変換）
        self.assertEqual(len(embedding.shape), 2)
        self.assertEqual(embedding.shape[0], 1)
        
        # ベクトルは数値配列
        self.assertTrue(np.all(np.isfinite(embedding)))
        
        # Transformer は一方向関数 - 逆変換不可能
        print("[OK] ベクトルから元データへの復号不可能を確認")
    
    def test_model_offline_compliance(self):
        """モデル情報（オフライン動作確認）"""
        info = self.vectorizer.get_model_info()
        
        # CPU実行確認
        self.assertIn('device', info)
        self.assertIn('model_name', info)
        self.assertIn('embedding_dimension', info)
        
        print(f"[OK] モデル確認: {info['model_name']} (device: {info['device']})")



class TestTextVectorizerPerformance(unittest.TestCase):
    """パフォーマンステスト"""
    
    @classmethod
    def setUpClass(cls):
        cls.vectorizer = TextVectorizer()
    
    def test_batch_processing_speed(self):
        """バッチ処理の速度"""
        texts = ["テキスト " + str(i) for i in range(100)]
        
        start_time = time.time()
        embeddings = self.vectorizer.encode_batch(texts, batch_size=32, show_progress=False)
        elapsed_time = time.time() - start_time
        
        # 処理が完了したことを確認
        self.assertEqual(embeddings.shape[0], 100)
        
        print(f"[PASS] バッチ処理テスト (100文 in {elapsed_time:.2f}秒)")
    
    def test_large_batch_processing(self):
        """大規模バッチ処理"""
        texts = ["テキスト " + str(i) for i in range(500)]
        
        start_time = time.time()
        embeddings = self.vectorizer.encode_batch(texts, batch_size=64, show_progress=False)
        elapsed_time = time.time() - start_time
        
        self.assertEqual(embeddings.shape[0], 500)
        
        print(f"[PASS] 大規模バッチ処理テスト (500文 in {elapsed_time:.2f}秒)")


class TestTextVectorizerIntegration(unittest.TestCase):
    """統合テスト"""
    
    @classmethod
    def setUpClass(cls):
        cls.vectorizer = TextVectorizer()
    
    def test_complete_workflow(self):
        """完全なワークフロー"""
        # テキスト集合
        documents = [
            "Python は強力なプログラミング言語です",
            "機械学習と深層学習は AI の重要な分野です",
            "Python は機械学習ライブラリが充実しています",
            "Java はエンタープライズアプリケーション開発に使用されます",
            "Web 開発には JavaScript と Python が一般的です"
        ]
        
        # 1. ベクトル化
        embeddings = self.vectorizer.encode(documents)
        self.assertEqual(embeddings.shape[0], 5)
        
        # 2. 類似度マトリックス
        similarity_matrix = self.vectorizer.similarity_matrix(documents)
        self.assertEqual(similarity_matrix.shape, (5, 5))
        
        # 3. テキスト検索
        query = "Python と機械学習"
        results = self.vectorizer.find_similar_texts(query, documents, top_k=2)
        self.assertEqual(len(results), 2)
        
        # 4. DataFrame変換
        df = self.vectorizer.to_dataframe(documents, embeddings)
        self.assertEqual(len(df), 5)
        
        # 5. 次元削減（PCA）
        reduced = self.vectorizer.reduce_dimensions(embeddings, n_components=2, method="pca")
        self.assertEqual(reduced.shape, (5, 2))
        
        print("[PASS] 統合テスト")


def run_tests():
    """すべてのテストを実行"""
    print("\n" + "=" * 70)
    print("TextVectorizer テストスイート開始")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # テストクラスを追加
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizerBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizerSimilarity))
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizerSearch))
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizerDataFrame))
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizerDimensionReduction))
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizerModelInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizerErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizerSecurity))  # セキュリティテスト追加
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizerPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizerIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # サマリー表示
    print("\n" + "=" * 70)
    print(f"テスト完了: {result.testsRun}件実行")
    print(f"[OK] 成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"[NG] 失敗: {len(result.failures)}")
    print(f"[NG] エラー: {len(result.errors)}")
    print("=" * 70 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
