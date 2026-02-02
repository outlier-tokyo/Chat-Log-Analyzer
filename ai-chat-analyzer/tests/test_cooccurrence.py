"""Test suite for CooccurrenceNetwork"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.cooccurrence import CooccurrenceNetwork


class TestCooccurrenceNetworkBasic:
    """Test basic functionality of CooccurrenceNetwork"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.network = CooccurrenceNetwork(window_size=3, min_frequency=1)
        self.tokenized_docs = [
            ['python', '機械学習', 'ライブラリ'],
            ['python', 'データ', '分析'],
            ['機械学習', 'アルゴリズム', 'データ'],
            ['ディープラーニング', 'ニューラルネットワーク', 'アルゴリズム']
        ]
    
    def test_initialization(self):
        """Test network initialization"""
        assert self.network is not None
        assert self.network.window_size == 3
        assert self.network.min_frequency == 1
    
    def test_build_network(self):
        """Test network building"""
        graph = self.network.build_network(self.tokenized_docs)
        assert graph is not None
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
    
    def test_network_has_nodes(self):
        """Test that network contains expected nodes"""
        graph = self.network.build_network(self.tokenized_docs)
        nodes = list(graph.nodes())
        assert 'python' in nodes
        assert 'データ' in nodes
        assert 'アルゴリズム' in nodes
    
    def test_network_structure(self):
        """Test network structure properties"""
        graph = self.network.build_network(self.tokenized_docs)
        
        # Check node count
        node_count = graph.number_of_nodes()
        assert node_count > 0
        
        # Check edge count
        edge_count = graph.number_of_edges()
        assert edge_count > 0
        
        # Edges should be less than or equal to possible edges
        max_edges = node_count * (node_count - 1) / 2
        assert edge_count <= max_edges


class TestCooccurrenceMatrixConversion:
    """Test co-occurrence matrix operations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.network = CooccurrenceNetwork(window_size=2, min_frequency=1)
        self.tokenized_docs = [
            ['a', 'b', 'c'],
            ['b', 'c', 'd'],
            ['a', 'd', 'e']
        ]
    
    def test_get_cooccurrence_dataframe(self):
        """Test co-occurrence dataframe creation"""
        graph = self.network.build_network(self.tokenized_docs)
        df = self.network.get_cooccurrence_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'word1' in df.columns
        assert 'word2' in df.columns
        assert 'frequency' in df.columns
    
    def test_cooccurrence_dataframe_values(self):
        """Test that cooccurrence dataframe contains valid data"""
        graph = self.network.build_network(self.tokenized_docs)
        df = self.network.get_cooccurrence_dataframe()
        
        # All frequencies should be positive
        assert (df['frequency'] > 0).all()
        
        # word1 and word2 should not be the same
        for idx, row in df.iterrows():
            assert row['word1'] != row['word2']


class TestCentralityMeasures:
    """Test network centrality measures"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.network = CooccurrenceNetwork(window_size=3, min_frequency=1)
        self.tokenized_docs = [
            ['word1', 'word2', 'word3'],
            ['word1', 'word2', 'word4'],
            ['word2', 'word3', 'word4'],
            ['word1', 'word3', 'word5']
        ]
        self.graph = self.network.build_network(self.tokenized_docs)
    
    def test_get_node_degree(self):
        """Test degree centrality calculation"""
        degree = self.network.get_node_degree()
        
        assert isinstance(degree, dict)
        assert len(degree) > 0
        assert all(isinstance(v, int) for v in degree.values())
    
    def test_degree_values_positive(self):
        """Test that degree values are positive"""
        degree = self.network.get_node_degree()
        assert all(v > 0 for v in degree.values())
    
    def test_get_node_strength(self):
        """Test strength centrality (weighted degree)"""
        strength = self.network.get_node_strength()
        
        assert isinstance(strength, dict)
        assert len(strength) > 0
        assert all(isinstance(v, (int, float)) for v in strength.values())
    
    def test_get_betweenness_centrality(self):
        """Test betweenness centrality"""
        betweenness = self.network.get_betweenness_centrality()
        
        assert isinstance(betweenness, dict)
        assert len(betweenness) > 0
        # Betweenness values are between 0 and 1 for normalized
        for v in betweenness.values():
            assert 0 <= v <= 1
    
    def test_get_closeness_centrality(self):
        """Test closeness centrality"""
        closeness = self.network.get_closeness_centrality()
        
        assert isinstance(closeness, dict)
        assert len(closeness) > 0
        # Closeness values are between 0 and 1
        for v in closeness.values():
            assert 0 <= v <= 1
    
    def test_get_eigenvector_centrality(self):
        """Test eigenvector centrality"""
        eigenvector = self.network.get_eigenvector_centrality()
        
        assert isinstance(eigenvector, dict)
        assert len(eigenvector) > 0


class TestKeywordExtraction:
    """Test keyword extraction functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.network = CooccurrenceNetwork(window_size=3, min_frequency=1)
        self.tokenized_docs = [
            ['python', 'programming', 'language'],
            ['python', 'machine', 'learning'],
            ['machine', 'learning', 'algorithm'],
            ['programming', 'language', 'python']
        ]
        self.graph = self.network.build_network(self.tokenized_docs)
    
    def test_get_keywords_by_frequency(self):
        """Test keyword extraction by frequency"""
        keywords = self.network.get_keywords_by_frequency(top_n=5)
        
        assert isinstance(keywords, pd.DataFrame)
        assert len(keywords) > 0
        assert 'word' in keywords.columns
        assert 'frequency' in keywords.columns
    
    def test_keywords_ordered_by_frequency(self):
        """Test that keywords are ordered by frequency"""
        keywords = self.network.get_keywords_by_frequency(top_n=10)
        
        frequencies = keywords['frequency'].tolist()
        assert frequencies == sorted(frequencies, reverse=True)
    
    def test_top_n_parameter(self):
        """Test that top_n parameter works correctly"""
        keywords_5 = self.network.get_keywords_by_frequency(top_n=5)
        keywords_3 = self.network.get_keywords_by_frequency(top_n=3)
        
        assert len(keywords_5) <= 5
        assert len(keywords_3) <= 3
    
    def test_get_top_edges(self):
        """Test top edges extraction"""
        if hasattr(self.network, 'get_top_edges'):
            edges = self.network.get_top_edges(top_n=5)
            
            assert isinstance(edges, pd.DataFrame)
            assert len(edges) > 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.network = CooccurrenceNetwork(window_size=2, min_frequency=1)
    
    def test_single_document(self):
        """Test with single document"""
        tokenized = [['word1', 'word2', 'word3']]
        graph = self.network.build_network(tokenized)
        
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
    
    def test_empty_document(self):
        """Test with empty documents"""
        tokenized = [[], ['a', 'b'], []]
        graph = self.network.build_network(tokenized)
        
        # Should handle empty docs gracefully
        assert graph.number_of_nodes() >= 0
    
    def test_single_token_per_document(self):
        """Test with single token per document"""
        tokenized = [['a'], ['b'], ['c'], ['d']]
        graph = self.network.build_network(tokenized)
        
        # No co-occurrences possible
        assert graph.number_of_edges() == 0
    
    def test_duplicate_words(self):
        """Test handling of duplicate words"""
        tokenized = [['word', 'word', 'word'], ['word', 'other']]
        graph = self.network.build_network(tokenized)
        
        # Should handle duplicates
        assert graph.number_of_nodes() > 0
    
    def test_large_document_set(self):
        """Test with large document set"""
        import random
        vocab = [f'word_{i}' for i in range(100)]
        tokenized = [random.sample(vocab, 10) for _ in range(50)]
        
        graph = self.network.build_network(tokenized)
        
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
    
    def test_min_frequency_filter(self):
        """Test minimum frequency filtering"""
        tokenized = [
            ['rare', 'common1', 'common2'],
            ['common1', 'common2', 'common3'],
            ['common1', 'common2', 'common3']
        ]
        
        # Network with min_frequency=1 should include all
        network1 = CooccurrenceNetwork(window_size=3, min_frequency=1)
        graph1 = network1.build_network(tokenized)
        
        # Network with min_frequency=2 should filter rare words
        network2 = CooccurrenceNetwork(window_size=3, min_frequency=2)
        graph2 = network2.build_network(tokenized)
        
        # graph2 should have fewer or equal nodes than graph1
        assert graph2.number_of_nodes() <= graph1.number_of_nodes()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
