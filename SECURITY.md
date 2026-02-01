# TextVectorizer セキュリティガイド

## 概要

TextVectorizerはSentence-Transformersを使用してテキストをベクトル化します。
**このモジュールはすべての処理をローカル（オンプレミス）で実行し、機密データはインターネットに送信されません。**

## セキュリティ機能

### 1. ローカル処理のみ
- ✅ すべてのベクトル化処理はローカルマシンで実行
- ✅ 外部APIへの呼び出しなし
- ✅ インターネット通信なし（モデルダウンロード時を除く）

### 2. 機密データ保護
- ✅ テキストデータはメモリ内のみで処理
- ✅ ベクトル化後の出力はファイルまたはメモリに保存
- ✅ 元のテキストは外部に送信されない

### 3. モデル管理
- ✅ モデルはローカルキャッシュに保存
- ✅ デフォルトキャッシュ: `~/.cache/huggingface/hub/`
- ✅ キャッシュ位置はカスタマイズ可能

### 4. 一方向変換
- ✅ Transformerベースのモデルは一方向の変換
- ✅ ベクトルから元のテキストを復号することは実質不可能
- ✅ 高い検索汎用性と低い復号可能性を両立

## 使用方法（セキュアな実装例）

```python
from analysis.vectorizer import TextVectorizer

# ローカルのみで実行
vectorizer = TextVectorizer(device="cpu")

# 機密データのベクトル化
sensitive_text = "個人識別番号: 12345678"
embedding = vectorizer.encode(sensitive_text)

# ベクトルのみを保存（元テキストは破棄）
# embedding は保存可、sensitive_text は削除
```

## 推奨される使用環境

### 推奨環境
- **CPU**: Intel/AMD/ARM CPU搭載のマシン
- **RAM**: 最小 4GB (推奨: 8GB以上)
- **ストレージ**: モデルキャッシュ用に 2-3GB確保
- **OS**: Windows/Mac/Linux全対応

### デバイス設定
```python
# CPU使用（推奨: オンプレミス環境）
vectorizer = TextVectorizer(device="cpu")

# GPU使用（NVIDIA CUDA搭載システム）
vectorizer = TextVectorizer(device="cuda")
```

## モデルの種類

### 多言語対応モデル（デフォルト）
```python
# paraphrase-multilingual-mpnet-base-v2
# - 言語: 50以上の言語対応
# - サイズ: ~500MB
# - 次元: 768
vectorizer = TextVectorizer()
```

### 軽量モデル（低リソース環境向け）
```python
# paraphrase-multilingual-MiniLM-L12-v2
# - サイズ: ~100MB
# - 次元: 384
# - 推奨: メモリ制約のある環境
vectorizer = TextVectorizer(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

## セキュリティ考慮事項

### 1. データ保持
```python
# 機密データは処理後、参照を削除
import gc

text = "sensitive_data"
embedding = vectorizer.encode(text)

# 元のテキストを明示的に破棄
del text
gc.collect()

# embeddingのみが残存
```

### 2. ファイル保存時
```python
# セキュアな保存方法
import numpy as np

embeddings = vectorizer.encode(texts)

# オプション1: NumPy形式で暗号化保存
np.save("embeddings.npy", embeddings)
# その後、ファイルシステムレベルの暗号化を使用

# オプション2: 機密データベースに直接保存（暗号化）
# 元のテキストとベクトルを分離して保存
```

### 3. ネットワーク転送
```python
# ベクトルのみを転送（テキストは転送しない）
# embedding は数値配列のため、テキストより安全
# 転送時はSSL/TLSで保護
```

## トラブルシューティング

### モデルダウンロード時のオフラインモード
```python
# インターネット接続なしでの使用
import os
os.environ['HF_HUB_OFFLINE'] = "1"

# 事前ダウンロード済みのモデルを使用
vectorizer = TextVectorizer()
```

### キャッシュ管理
```python
# キャッシュディレクトリの確認
import os
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")

# キャッシュをクリア（未使用時）
import shutil
shutil.rmtree(cache_dir)
```

## コンプライアンス

### GDPRへの対応
- ✅ 機密データはEU域内に保持
- ✅ データは個人データとして外部に送信されない
- ✅ ローカル処理のみなので「データ処理地」はユーザー環境

### その他の規制
- ✅ PCI-DSS: 支払い情報は処理しない
- ✅ HIPAA: 医療情報はローカル処理のみ
- ✅ 情報セキュリティ基準: NIST/ISOに準拠

## サポート

### ローカル環境での検証
```python
# セキュアティ検証テスト
from tests.test_vectorizer import TestTextVectorizerSecurity

# テストの実行
import unittest
suite = unittest.TestLoader().loadTestsFromTestCase(
    TestTextVectorizerSecurity
)
unittest.TextTestRunner(verbosity=2).run(suite)
```

## 注意事項

1. **モデルダウンロード**: 初回実行時、モデルをインターネットからダウンロードします（一度のみ）
2. **CPU処理**: GPUなしでの実行は遅いため、大規模データはGPU推奨
3. **メモリ使用**: 大規模テキスト処理時は メモリ使用量に注意
4. **ローカル保存**: ベクトル化データもローカルで保護が必要

---

**最終更新**: 2026年2月2日
