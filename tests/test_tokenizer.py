#!/usr/bin/env python3
"""
Test script for Tokenizer.

This script tests the Tokenizer implementation with various Japanese text samples.

Usage:
    python test_tokenizer.py
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "ai-chat-analyzer" / "src"
sys.path.insert(0, str(src_path))

try:
    from preprocessor.tokenizer import Tokenizer
except ImportError as e:
    print(f"Error: Could not import Tokenizer from {src_path}")
    print(f"Details: {e}")
    sys.exit(1)


def test_basic_tokenization():
    """Test basic tokenization."""
    print("=" * 70)
    print("Testing Tokenizer")
    print("=" * 70)
    
    print("\n1. Basic Tokenization Test:")
    print("-" * 70)
    
    try:
        tokenizer = Tokenizer()
        
        # Test cases
        test_cases = [
            "こんにちは",
            "今日の天気は晴れです",
            "お疲れ様です。本日の会議は14時00分です。",
            "自然言語処理について学習しています",
        ]
        
        for i, text in enumerate(test_cases, 1):
            tokens = tokenizer.tokenize(text)
            print(f"\n  Test {i}: {text}")
            print(f"  Tokens: {tokens}")
            print(f"  Count: {len(tokens)}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_tokenize_with_pos():
    """Test tokenization with POS tags."""
    print("\n2. Tokenization with POS Tags:")
    print("-" * 70)
    
    try:
        tokenizer = Tokenizer()
        text = "今日の天気は晴れです"
        
        tokens_with_pos = tokenizer.tokenize_with_pos(text)
        
        print(f"\n  Text: {text}")
        print(f"  {'Token':<12} {'POS':<10} {'POS1':<12} {'Base Form':<12}")
        print(f"  {'-'*50}")
        for token_info in tokens_with_pos:
            print(f"  {token_info['token']:<12} {token_info['pos']:<10} "
                  f"{token_info['pos1']:<12} {token_info['base']:<12}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_base_forms():
    """Test extraction of base forms."""
    print("\n3. Base Form Extraction:")
    print("-" * 70)
    
    try:
        tokenizer = Tokenizer()
        text = "走って、書いた、見ている"
        
        surface_forms = tokenizer.tokenize(text)
        base_forms = tokenizer.tokenize_base_forms(text)
        
        print(f"\n  Text: {text}")
        print(f"  Surface forms: {surface_forms}")
        print(f"  Base forms:    {base_forms}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_pos_filtering():
    """Test POS tag filtering."""
    print("\n4. POS Tag Filtering:")
    print("-" * 70)
    
    try:
        tokenizer = Tokenizer()
        text = "自然言語処理について学習しています"
        
        # Extract nouns
        nouns = tokenizer.tokenize_nouns(text)
        print(f"\n  Text: {text}")
        print(f"  Nouns (base form):     {nouns}")
        
        # Extract verbs
        verbs = tokenizer.tokenize_verbs(text)
        print(f"  Verbs (base form):     {verbs}")
        
        # Extract adjectives
        adjectives = tokenizer.tokenize_adjectives(text)
        print(f"  Adjectives (base form): {adjectives}")
        
        # Custom filtering
        filtered = tokenizer.tokenize_filtered(
            text,
            pos_tags=['名詞', '動詞'],
            use_base_form=True
        )
        print(f"  Nouns + Verbs:         {filtered}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_pos_statistics():
    """Test POS statistics."""
    print("\n5. POS Tag Statistics:")
    print("-" * 70)
    
    try:
        tokenizer = Tokenizer()
        text = "自然言語処理についての学習を始めました"
        
        stats = tokenizer.get_pos_statistics(text)
        
        print(f"\n  Text: {text}")
        print(f"  POS Tag Distribution:")
        for pos, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"    {pos:<15}: {count}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_real_world_examples():
    """Test with real-world examples."""
    print("\n6. Real-World Examples:")
    print("-" * 70)
    
    try:
        tokenizer = Tokenizer()
        
        examples = [
            "こんにちは、今日はいい天気ですね。",
            "自然言語処理の技術を使ってテキスト分析を行います。",
            "お疲れ様です。本日の会議は予定通り開催します。",
            "このシステムは複数の言語に対応しています。",
        ]
        
        for i, text in enumerate(examples, 1):
            tokens = tokenizer.tokenize(text)
            nouns = tokenizer.tokenize_nouns(text)
            
            print(f"\n  Example {i}: {text}")
            print(f"    All tokens ({len(tokens)}):  {tokens}")
            print(f"    Nouns ({len(nouns)}):        {nouns}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_edge_cases():
    """Test edge cases."""
    print("\n7. Edge Cases:")
    print("-" * 70)
    
    try:
        tokenizer = Tokenizer()
        
        edge_cases = [
            ("", "Empty string"),
            ("   ", "Whitespace only"),
            ("123", "Numbers only"),
            ("ABC", "ASCII letters"),
            ("😀絵文字", "Emoji"),
        ]
        
        for text, description in edge_cases:
            try:
                tokens = tokenizer.tokenize(text)
                print(f"\n  {description}: {repr(text)}")
                print(f"    Tokens: {tokens}")
            except Exception as e:
                print(f"\n  {description}: {repr(text)}")
                print(f"    [ERROR] {e}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_performance():
    """Test performance with large text."""
    print("\n8. Performance Test:")
    print("-" * 70)
    
    try:
        tokenizer = Tokenizer()
        
        # Generate large text
        large_text = "これはテストテキストです。" * 100
        
        import time
        
        start = time.time()
        tokens = tokenizer.tokenize(large_text)
        elapsed = time.time() - start
        
        print(f"\n  Input size: {len(large_text)} characters")
        print(f"  Token count: {len(tokens)}")
        print(f"  Processing time: {elapsed*1000:.2f}ms")
        print(f"  Status: [PASS]" if elapsed < 2.0 else f"  Status: [WARNING] (slow)")
        
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    """Run all tests."""
    try:
        results = []
        
        results.append(("Basic Tokenization", test_basic_tokenization()))
        results.append(("POS Tags", test_tokenize_with_pos()))
        results.append(("Base Forms", test_base_forms()))
        results.append(("POS Filtering", test_pos_filtering()))
        results.append(("POS Statistics", test_pos_statistics()))
        results.append(("Real-World Examples", test_real_world_examples()))
        results.append(("Edge Cases", test_edge_cases()))
        results.append(("Performance", test_performance()))
        
        print("\n" + "=" * 70)
        print("Test Summary:")
        print("-" * 70)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {name:<30} {status}")
        
        print("-" * 70)
        print(f"  Total: {passed}/{total} passed")
        print("=" * 70)
        
        return passed == total
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
