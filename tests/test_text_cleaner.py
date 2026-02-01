#!/usr/bin/env python3
"""
Test script for TextCleaner.

This script tests the TextCleaner implementation with various text samples.

Usage:
    python test_text_cleaner.py
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "ai-chat-analyzer" / "src"
sys.path.insert(0, str(src_path))

try:
    from preprocessor.text_cleaner import TextCleaner
except ImportError as e:
    print(f"Error: Could not import TextCleaner from {src_path}")
    print(f"Details: {e}")
    sys.exit(1)


def test_basic_cleaning():
    """Test basic text cleaning operations."""
    print("=" * 70)
    print("Testing TextCleaner")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "HTML tags removal",
            "input": "<p>こんにちは<br/>世界</p>",
            "expected": "こんにちは世界"
        },
        {
            "name": "URL removal",
            "input": "詳細はhttps://example.com を参照してください",
            "expected": "詳細は を参照してください"
        },
        {
            "name": "Email removal",
            "input": "連絡先はuser@example.com までお願いします",
            "expected": "までお願いします"
        },
        {
            "name": "Whitespace normalization",
            "input": "これは   複数の\n\n空白を含んでいます",
            "expected": "これは 複数の 空白を含んでいます"
        },
        {
            "name": "Full-width space to half-width",
            "input": "全角　スペース　を　含んでいます",
            "expected": "全角 スペース を 含んでいます"
        },
        {
            "name": "Leading/trailing whitespace",
            "input": "  　前後に空白がある　  ",
            "expected": "前後に空白がある"
        },
        {
            "name": "Combined cleaning",
            "input": "  <p>HTMLと https://url.com  \n\n複数空白</p>  ",
            "expected": "HTMLと 複数空白"
        },
        {
            "name": "Non-string input",
            "input": 12345,
            "expected": ""
        },
        {
            "name": "None input",
            "input": None,
            "expected": ""
        },
        {
            "name": "Empty string",
            "input": "",
            "expected": ""
        }
    ]
    
    print("\n1. Basic Cleaning Tests:")
    print("-" * 70)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        result = TextCleaner.clean(test_case["input"])
        is_pass = result == test_case["expected"]
        
        status = "PASS" if is_pass else "FAIL"
        print(f"\n  Test {i}: {test_case['name']}")
        print(f"  Input:    {repr(test_case['input'])}")
        print(f"  Expected: {repr(test_case['expected'])}")
        print(f"  Got:      {repr(result)}")
        print(f"  [{status}]")
        
        if is_pass:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 70)
    print(f"  Results: {passed} passed, {failed} failed")
    
    return failed == 0


def test_custom_options():
    """Test text cleaning with custom options."""
    print("\n2. Custom Options Tests:")
    print("-" * 70)
    
    test_cases = [
        {
            "name": "Keep URLs",
            "input": "詳細はhttps://example.com を参照",
            "options": {"remove_urls": False},
            "expected": "詳細はhttps://example.com を参照"
        },
        {
            "name": "Remove numbers",
            "input": "年齢は25歳で、TEL: 090-1234-5678です",
            "options": {"remove_numbers": True},
            "expected": "年齢は歳で、TEL: --です"
        },
        {
            "name": "Keep HTML",
            "input": "<b>太字</b>のテキスト",
            "options": {"remove_html": False},
            "expected": "<b>太字</b>のテキスト"
        },
        {
            "name": "Keep whitespace",
            "input": "複数   空白\n\n改行",
            "options": {"collapse_whitespace": False},
            "expected": "複数   空白\n\n改行"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        result = TextCleaner.clean_with_options(
            test_case["input"],
            **test_case["options"]
        )
        is_pass = result == test_case["expected"]
        
        status = "PASS" if is_pass else "FAIL"
        print(f"\n  Test {i}: {test_case['name']}")
        print(f"  Input:    {repr(test_case['input'])}")
        print(f"  Options:  {test_case['options']}")
        print(f"  Expected: {repr(test_case['expected'])}")
        print(f"  Got:      {repr(result)}")
        print(f"  [{status}]")
        
        if is_pass:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 70)
    print(f"  Results: {passed} passed, {failed} failed")
    
    return failed == 0


def test_real_world_examples():
    """Test with realistic chat message examples."""
    print("\n3. Real-World Examples:")
    print("-" * 70)
    
    examples = [
        "こんにちは、元気ですか？",
        "<p>お疲れ様です。</p><br>本日の会議は14:00です。",
        "詳細は https://docs.example.com/guide を確認してください",
        "お問い合わせは contact@example.com までお願いします",
        "年代：30代　性別：男性　",
        "  　前後に余計なスペースがあります　  ",
        "複数の\n\n\n改行を含んでいます",
        "😀絵文字も含まれています😊",
        "全角ＡＢＣ数字１２３が含まれています",
    ]
    
    for i, example in enumerate(examples, 1):
        cleaned = TextCleaner.clean(example)
        print(f"\n  Example {i}:")
        print(f"    Input:  {repr(example)}")
        print(f"    Output: {repr(cleaned)}")


def test_performance():
    """Test performance with large text."""
    print("\n4. Performance Test:")
    print("-" * 70)
    
    # Generate large text
    large_text = " ".join(["これはテストテキストです"] * 1000)
    
    import time
    
    start = time.time()
    result = TextCleaner.clean(large_text)
    elapsed = time.time() - start
    
    print(f"  Input size: {len(large_text)} characters")
    print(f"  Output size: {len(result)} characters")
    print(f"  Processing time: {elapsed*1000:.2f}ms")
    print(f"  Status: [PASS]" if elapsed < 1.0 else f"  Status: [WARNING] (slow)")


def main():
    """Run all tests."""
    try:
        basic_pass = test_basic_cleaning()
        custom_pass = test_custom_options()
        test_real_world_examples()
        test_performance()
        
        print("\n" + "=" * 70)
        if basic_pass and custom_pass:
            print("[SUCCESS] All critical tests passed!")
            print("=" * 70)
            return True
        else:
            print("[FAILURE] Some tests failed")
            print("=" * 70)
            return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
