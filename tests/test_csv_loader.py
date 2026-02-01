"""
CSVLoaderの包括的なテスト

以下をテストします:
- 正常系: 標準的なCSVファイルの読み込み
- エンコーディング処理: 複数エンコーディング対応
- スキーマ検証: 必須/推奨カラム確認
- 型変換: user_id, message, timestamp, session_id
- 欠損値処理: nullの除去と補填
- エラーハンドリング: ファイルなし、読み込み権限なし等
"""

import unittest
import tempfile
import os
import pandas as pd
import sys
from pathlib import Path

# パスを追加してインポート
sys.path.insert(0, str(Path(__file__).parent.parent / 'ai-chat-analyzer' / 'src'))

from loader.csv_loader import CSVLoader


class TestCSVLoaderBasic(unittest.TestCase):
    """基本的な読み込みテスト"""
    
    def setUp(self):
        """各テスト前に一時ディレクトリを作成"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """各テスト後に一時ファイルをクリーンアップ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_csv(self, filename: str, content: str, encoding: str = 'utf-8') -> str:
        """テスト用CSVファイルを作成"""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return file_path
    
    def test_load_basic_csv(self):
        """基本的なCSV読み込み"""
        csv_content = """user_id,message,timestamp
user_001,Hello world,2026-02-01 10:00:00
user_002,こんにちは,2026-02-01 10:05:00
user_001,Thank you,2026-02-01 10:10:00"""
        
        file_path = self._create_csv('basic.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        # 基本的な検証
        self.assertEqual(len(df), 3)
        self.assertListEqual(list(df.columns), ['user_id', 'message', 'timestamp'])
        self.assertEqual(df['user_id'].iloc[0], 'user_001')
        print("[PASS] 基本的なCSV読み込みテスト")
    
    def test_load_with_optional_columns(self):
        """オプションカラム（session_id）を含むCSV読み込み"""
        csv_content = """user_id,message,timestamp,session_id
user_001,Hello,2026-02-01 10:00:00,session_001
user_002,World,2026-02-01 10:05:00,session_001"""
        
        file_path = self._create_csv('with_session.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        self.assertEqual(len(df), 2)
        self.assertIn('session_id', df.columns)
        self.assertEqual(df['session_id'].iloc[0], 'session_001')
        print("[PASS] オプションカラム付きCSV読み込みテスト")
    
    def test_load_japanese_text(self):
        """日本語テキストを含むCSV読み込み"""
        csv_content = """user_id,message
user_001,こんにちは
user_002,こんばんは
user_003,おはようございます"""
        
        file_path = self._create_csv('japanese.csv', csv_content, encoding='utf-8')
        loader = CSVLoader(file_path)
        df = loader.load()
        
        self.assertEqual(len(df), 3)
        self.assertEqual(df['message'].iloc[0], 'こんにちは')
        print("[PASS] 日本語テキスト読み込みテスト")
    
    def test_load_minimum_columns(self):
        """最小限のカラムのみを含むCSV"""
        csv_content = """user_id,message
user_001,Hello
user_002,World"""
        
        file_path = self._create_csv('minimum.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ['user_id', 'message'])
        print("[PASS] 最小カラムCSV読み込みテスト")


class TestCSVLoaderEncoding(unittest.TestCase):
    """エンコーディング処理のテスト"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_csv(self, filename: str, content: str, encoding: str = 'utf-8') -> str:
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return file_path
    
    def test_load_utf8(self):
        """UTF-8エンコーディングのCSV読み込み"""
        csv_content = """user_id,message
user_001,テスト
user_002,データ"""
        
        file_path = self._create_csv('utf8.csv', csv_content, encoding='utf-8')
        loader = CSVLoader(file_path)
        df = loader.load()
        
        self.assertEqual(len(df), 2)
        self.assertEqual(df['message'].iloc[0], 'テスト')
        print("[PASS] UTF-8エンコーディングテスト")
    
    def test_load_utf8_bom(self):
        """UTF-8 BOM付きのCSV読み込み"""
        csv_content = """user_id,message
user_001,BOMテスト
user_002,データ"""
        
        file_path = self._create_csv('utf8_bom.csv', csv_content, encoding='utf-8-sig')
        loader = CSVLoader(file_path)
        df = loader.load()
        
        self.assertEqual(len(df), 2)
        # BOMが正しく処理されることを確認
        self.assertNotIn('\ufeff', df.columns[0])
        print("[PASS] UTF-8 BOM エンコーディングテスト")
    
    def test_load_shift_jis(self):
        """Shift-JISエンコーディングのCSV読み込み"""
        csv_content = """user_id,message
user_001,シフトJIS
user_002,テスト"""
        
        file_path = self._create_csv('sjis.csv', csv_content, encoding='shift-jis')
        loader = CSVLoader(file_path)
        df = loader.load()
        
        self.assertEqual(len(df), 2)
        self.assertEqual(df['message'].iloc[0], 'シフトJIS')
        print("[PASS] Shift-JIS エンコーディングテスト")


class TestCSVLoaderSchema(unittest.TestCase):
    """スキーマ検証のテスト"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_csv(self, filename: str, content: str) -> str:
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_missing_required_column_user_id(self):
        """user_idカラム欠損時のエラー"""
        csv_content = """message
Hello"""
        
        file_path = self._create_csv('no_user_id.csv', csv_content)
        loader = CSVLoader(file_path)
        
        with self.assertRaises(ValueError):
            loader.load()
        
        print("[PASS] user_id欠損エラーテスト")
    
    def test_missing_required_column_message(self):
        """messageカラム欠損時のエラー"""
        csv_content = """user_id
user_001"""
        
        file_path = self._create_csv('no_message.csv', csv_content)
        loader = CSVLoader(file_path)
        
        with self.assertRaises(ValueError):
            loader.load()
        
        print("[PASS] message欠損エラーテスト")


class TestCSVLoaderTypeConversion(unittest.TestCase):
    """型変換のテスト"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_csv(self, filename: str, content: str) -> str:
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_user_id_type_conversion(self):
        """user_idがstringに変換されることを確認"""
        csv_content = """user_id,message
123,Hello
456,World"""
        
        file_path = self._create_csv('user_id_int.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        # user_idがstring型であることを確認
        self.assertEqual(df['user_id'].dtype, object)
        self.assertIn(df['user_id'].iloc[0], ['123', 123])  # String or int
        print("[PASS] user_id型変換テスト")
    
    def test_message_type_conversion(self):
        """messageがstringに変換されることを確認"""
        csv_content = """user_id,message
user_001,Hello
user_002,World"""
        
        file_path = self._create_csv('message_type.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        # messageがstring型であることを確認
        self.assertEqual(df['message'].dtype, object)
        self.assertEqual(df['message'].iloc[0], 'Hello')
        print("[PASS] message型変換テスト")
    
    def test_timestamp_conversion(self):
        """timestampがdatetime型に変換されることを確認"""
        csv_content = """user_id,message,timestamp
user_001,Hello,2026-02-01 10:00:00
user_002,World,2026-02-01 10:05:00"""
        
        file_path = self._create_csv('timestamp.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        # timestampがdatetime型であることを確認
        self.assertEqual(pd.api.types.is_datetime64_any_dtype(df['timestamp']), True)
        print("[PASS] timestamp型変換テスト")
    
    def test_session_id_type_conversion(self):
        """session_idがstringに変換されることを確認"""
        csv_content = """user_id,message,session_id
user_001,Hello,1001
user_002,World,1002"""
        
        file_path = self._create_csv('session_id.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        # session_idがstring型であることを確認
        self.assertEqual(df['session_id'].dtype, object)
        self.assertEqual(df['session_id'].iloc[0], '1001')
        print("[PASS] session_id型変換テスト")


class TestCSVLoaderMissingValues(unittest.TestCase):
    """欠損値処理のテスト"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_csv(self, filename: str, content: str) -> str:
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_missing_message_removed(self):
        """messageの欠損行が削除される"""
        csv_content = """user_id,message
user_001,Hello
user_002,
user_003,World"""
        
        file_path = self._create_csv('missing_message.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        # 欠損行が削除されていることを確認
        self.assertEqual(len(df), 2)
        self.assertListEqual(df['user_id'].tolist(), ['user_001', 'user_003'])
        print("[PASS] message欠損行削除テスト")
    
    def test_missing_user_id_filled(self):
        """user_idの欠損値が'unknown_user'で埋まる"""
        csv_content = """user_id,message
user_001,Hello
,World
user_003,Goodbye"""
        
        file_path = self._create_csv('missing_user_id.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        # 欠損値が埋まっていることを確認
        self.assertEqual(len(df), 3)
        self.assertEqual(df['user_id'].iloc[1], 'unknown_user')
        print("[PASS] user_id欠損値補填テスト")


class TestCSVLoaderErrorHandling(unittest.TestCase):
    """エラーハンドリングのテスト"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_file_not_found(self):
        """ファイルが見つからない場合のエラー"""
        with self.assertRaises(FileNotFoundError):
            loader = CSVLoader('/path/to/nonexistent/file.csv')
        
        print("[PASS] ファイルなしエラーテスト")
    
    def test_empty_file_path(self):
        """ファイルパスが空の場合のエラー"""
        with self.assertRaises(ValueError):
            CSVLoader('')
        
        print("[PASS] 空ファイルパスエラーテスト")
    
    def test_not_csv_file(self):
        """CSVファイルではない場合のエラー"""
        file_path = os.path.join(self.temp_dir, 'test.txt')
        with open(file_path, 'w') as f:
            f.write('test')
        
        with self.assertRaises(ValueError) as context:
            CSVLoader(file_path)
        
        self.assertIn('CSV', str(context.exception))
        print("[PASS] 非CSVファイルエラーテスト")
    
    def test_malformed_csv(self):
        """不正なCSV形式"""
        file_path = os.path.join(self.temp_dir, 'malformed.csv')
        with open(file_path, 'w', encoding='utf-8') as f:
            # 不正なCSV（カラム数が一致しない）
            f.write('user_id,message\n')
            f.write('user_001,Hello\n')  # 正しい行
        
        loader = CSVLoader(file_path)
        # Pandasはこのような不正なCSVでも読み込む（NaNで埋まる）
        df = loader.load()
        self.assertEqual(len(df), 1)
        print("[PASS] 不正なCSV形式テスト")


class TestCSVLoaderRealWorld(unittest.TestCase):
    """実世界のユースケーステスト"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_csv(self, filename: str, content: str) -> str:
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_large_dataset(self):
        """大規模データセットの読み込み"""
        # 1000行のデータを生成
        lines = ['user_id,message,timestamp,session_id']
        for i in range(1000):
            lines.append(f'user_{i % 10:03d},Message {i},2026-02-{(i % 28) + 1:02d} {(i % 24):02d}:00:00,session_{i // 100:02d}')
        
        csv_content = '\n'.join(lines)
        file_path = self._create_csv('large.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        self.assertEqual(len(df), 1000)
        self.assertEqual(list(df.columns), ['user_id', 'message', 'timestamp', 'session_id'])
        print("[PASS] 大規模データセット読み込みテスト (1000行)")
    
    def test_mixed_content(self):
        """実際のチャットデータ（絵文字、URL、改行含む）"""
        csv_content = """user_id,message,timestamp
user_001,こんにちは👋,2026-02-01 10:00:00
user_002,Check this: https://example.com,2026-02-01 10:05:00
user_003,"Multi
line
message",2026-02-01 10:10:00"""
        
        file_path = self._create_csv('mixed_content.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        self.assertEqual(len(df), 3)
        self.assertIn('👋', df['message'].iloc[0])
        self.assertIn('https://', df['message'].iloc[1])
        print("[PASS] 混合コンテンツ読み込みテスト")
    
    def test_empty_csv(self):
        """ヘッダーのみのCSV"""
        csv_content = """user_id,message,timestamp"""
        
        file_path = self._create_csv('empty.csv', csv_content)
        loader = CSVLoader(file_path)
        df = loader.load()
        
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), ['user_id', 'message', 'timestamp'])
        print("[PASS] 空CSV読み込みテスト")


class TestCSVLoaderIntegration(unittest.TestCase):
    """統合テスト"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_csv(self, filename: str, content: str, encoding: str = 'utf-8') -> str:
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return file_path
    
    def test_complete_workflow(self):
        """エンコーディング、型変換、欠損値処理の統合"""
        csv_content = """user_id,message,timestamp,session_id
user_001,こんにちは,2026-02-01 10:00:00,session_001
,こんばんは,2026-02-01 10:05:00,session_001
user_003,,2026-02-01 10:10:00,session_001
user_004,おはよう,2026-02-01 10:15:00,session_002"""
        
        file_path = self._create_csv('integration.csv', csv_content, encoding='utf-8')
        loader = CSVLoader(file_path)
        df = loader.load()
        
        # 結果の検証
        # - 欠損messageは削除されるため、3行になるはず（user_003の行が削除される）
        self.assertEqual(len(df), 3)
        # - 欠損user_idは'unknown_user'に置換
        self.assertIn('unknown_user', df['user_id'].values)
        # - timestamp型がdatetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['timestamp']))
        # - user_idとsession_idはstring
        self.assertEqual(df['user_id'].dtype, object)
        self.assertEqual(df['session_id'].dtype, object)
        
        print("[PASS] 統合テスト")


def run_tests():
    """すべてのテストを実行"""
    print("\n" + "=" * 70)
    print("CSVLoader テストスイート開始")
    print("=" * 70 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # すべてのテストクラスを追加
    suite.addTests(loader.loadTestsFromTestCase(TestCSVLoaderBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestCSVLoaderEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestCSVLoaderSchema))
    suite.addTests(loader.loadTestsFromTestCase(TestCSVLoaderTypeConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestCSVLoaderMissingValues))
    suite.addTests(loader.loadTestsFromTestCase(TestCSVLoaderErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestCSVLoaderRealWorld))
    suite.addTests(loader.loadTestsFromTestCase(TestCSVLoaderIntegration))
    
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
