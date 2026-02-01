# Tests Directory

This directory contains comprehensive tests for all modules in the Chat Log Analyzer project.

## Test Files

### Data Loaders
- `test_huggingface_loader.py` - Tests for HuggingFaceLoader with mock data generation

### Preprocessors
- `test_text_cleaner.py` - Tests for TextCleaner (HTML removal, URL/email removal, unicode normalization)
- `test_tokenizer.py` - Tests for Tokenizer (morphological analysis, POS filtering, base forms)

### Analysis Engines
- `test_clustering.py` - Tests for TopicClusterer (when implemented)
- `test_cooccurrence.py` - Tests for CooccurrenceNetwork (when implemented)
- `test_vectorizer.py` - Tests for TextVectorizer (when implemented)

### Visualization
- `test_charts.py` - Tests for visualization charts (when implemented)

## Running Tests

### Run all tests
```bash
cd c:\DEV\Chat-Log-Analyzer
python -m pytest tests/ -v
```

### Run specific test file
```bash
python tests/test_text_cleaner.py
python tests/test_tokenizer.py
```

### Run with coverage
```bash
python -m pytest tests/ --cov=ai-chat-analyzer/src
```

## Test Status

| Module | Status | Test File |
|--------|--------|-----------|
| HuggingFaceLoader | ✓ Complete | test_huggingface_loader.py |
| TextCleaner | ✓ Complete | test_text_cleaner.py |
| Tokenizer | ✓ Complete | test_tokenizer.py |
| CSVLoader | In Progress | test_csv_loader.py |
| TopicClusterer | Not Started | test_clustering.py |
| CooccurrenceNetwork | Not Started | test_cooccurrence.py |
| TextVectorizer | Not Started | test_vectorizer.py |
| Charts | Not Started | test_charts.py |

## Adding New Tests

When implementing new modules, create corresponding test files following the naming convention:
- `test_<module_name>.py`

Each test file should include:
1. Basic functionality tests
2. Edge case tests
3. Performance tests
4. Real-world example tests
5. Error handling tests
