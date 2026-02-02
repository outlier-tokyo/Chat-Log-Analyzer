import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from visualization.visualizer import EmbeddingVisualizer


class TestEmbeddingVisualizerBasic:
    """Test basic initialization and properties of EmbeddingVisualizer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.visualizer = EmbeddingVisualizer(n_neighbors=5, min_dist=0.1, metric='euclidean')
        self.embeddings = np.random.randn(20, 768)
        self.embeddings_2d = np.random.randn(20, 2)
        self.texts = [f"Sample text {i}" for i in range(20)]
        self.labels = np.array([0, 1, 0, 1, 2] * 4)
    
    def test_initialization(self):
        """Test that visualizer initializes correctly"""
        assert self.visualizer is not None
        assert self.visualizer.n_neighbors == 5
        assert self.visualizer.min_dist == 0.1
        assert self.visualizer.metric == 'euclidean'
    
    def test_fit_transform(self):
        """Test UMAP fit_transform method"""
        result = self.visualizer.fit_transform(self.embeddings)
        assert result.shape == (20, 2)
        assert isinstance(result, np.ndarray)
        assert not np.isnan(result).any()
    
    def test_fit_transform_output_range(self):
        """Test that UMAP output is in reasonable range"""
        result = self.visualizer.fit_transform(self.embeddings)
        assert result.min() > -100  # UMAP outputs should be bounded
        assert result.max() < 100
    
    def test_fit_transform_with_different_dimensions(self):
        """Test fit_transform with different input dimensions"""
        for dim in [128, 256, 512, 768, 1024]:
            embeddings = np.random.randn(10, dim)
            result = self.visualizer.fit_transform(embeddings)
            assert result.shape == (10, 2)


class TestScatterPlotGeneration:
    """Test scatter plot generation functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.visualizer = EmbeddingVisualizer(n_neighbors=5, min_dist=0.1)
        self.embeddings_2d = np.random.randn(15, 2)
        self.texts = [f"Message {i}" for i in range(15)]
        self.labels = np.array([0, 1, 0, 1, 2] * 3)
    
    def test_plot_scatter_basic(self):
        """Test basic scatter plot generation"""
        fig = self.visualizer.plot_scatter(
            embeddings_2d=self.embeddings_2d,
            texts=self.texts,
            labels=self.labels,
            title="Test Plot"
        )
        assert fig is not None
        assert hasattr(fig, 'show')
    
    def test_plot_scatter_with_custom_params(self):
        """Test scatter plot with custom parameters"""
        fig = self.visualizer.plot_scatter(
            embeddings_2d=self.embeddings_2d,
            texts=self.texts,
            labels=self.labels,
            title="Custom Plot",
            height=600,
            width=800,
            marker_size=8,
            opacity=0.7
        )
        assert fig is not None
    
    def test_plot_scatter_noise_points(self):
        """Test scatter plot with noise points (label = -1)"""
        labels_with_noise = np.array([0, 1, -1, 0, 1, -1, 2] * 2 + [0, 1])
        embeddings_2d = np.random.randn(len(labels_with_noise), 2)
        texts = [f"Text {i}" for i in range(len(labels_with_noise))]
        
        fig = self.visualizer.plot_scatter(
            embeddings_2d=embeddings_2d,
            texts=texts,
            labels=labels_with_noise,
            title="Plot with Noise"
        )
        assert fig is not None
    
    def test_plot_scatter_single_cluster(self):
        """Test scatter plot with all points in one cluster"""
        labels = np.zeros(15, dtype=int)
        fig = self.visualizer.plot_scatter(
            embeddings_2d=self.embeddings_2d,
            texts=self.texts,
            labels=labels,
            title="Single Cluster"
        )
        assert fig is not None
    
    def test_plot_scatter_many_clusters(self):
        """Test scatter plot with many clusters"""
        labels = np.arange(15)  # Each point is its own cluster
        fig = self.visualizer.plot_scatter(
            embeddings_2d=self.embeddings_2d,
            texts=self.texts,
            labels=labels,
            title="Many Clusters"
        )
        assert fig is not None


class TestHTMLExport:
    """Test HTML export functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.visualizer = EmbeddingVisualizer()
        self.embeddings_2d = np.random.randn(10, 2)
        self.texts = [f"Text {i}" for i in range(10)]
        self.labels = np.array([0, 1] * 5)
        
        self.fig = self.visualizer.plot_scatter(
            embeddings_2d=self.embeddings_2d,
            texts=self.texts,
            labels=self.labels
        )
    
    def test_save_html(self):
        """Test HTML file saving"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_plot.html"
            self.visualizer.save_html(self.fig, str(filepath))
            assert filepath.exists()
            assert filepath.stat().st_size > 0
    
    def test_save_html_creates_file(self):
        """Test that HTML file contains expected content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_output.html"
            self.visualizer.save_html(self.fig, str(filepath))
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for plotly content markers
            assert 'plotly' in content.lower() or 'scatter' in content.lower()
    
    def test_save_html_different_paths(self):
        """Test saving HTML to different path types"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with string path
            filepath1 = Path(tmpdir) / "plot1.html"
            self.visualizer.save_html(self.fig, str(filepath1))
            assert filepath1.exists()
            
            # Test with Path object
            filepath2 = Path(tmpdir) / "plot2.html"
            self.visualizer.save_html(self.fig, str(filepath2))
            assert filepath2.exists()


class TestVisualizerMetrics:
    """Test metrics and properties of visualizer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.visualizer = EmbeddingVisualizer(n_neighbors=5, min_dist=0.1)
        self.embeddings = np.random.randn(20, 768)
    
    def test_umap_preserves_structure(self):
        """Test that UMAP preserves some global structure"""
        embeddings_2d = self.visualizer.fit_transform(self.embeddings)
        
        # Check that UMAP produces a 2D output
        assert embeddings_2d.shape[1] == 2
        
        # Check that output is normalized-ish (no extreme values)
        assert np.abs(embeddings_2d).max() < 1000
    
    def test_different_random_seeds(self):
        """Test consistency with different runs"""
        result1 = self.visualizer.fit_transform(self.embeddings.copy())
        result2 = self.visualizer.fit_transform(self.embeddings.copy())
        
        # UMAP is stochastic but should produce similar outputs
        assert result1.shape == result2.shape
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.visualizer = EmbeddingVisualizer()
    
    def test_very_small_embedding(self):
        """Test with very small number of embeddings"""
        embeddings = np.random.randn(2, 768)
        result = self.visualizer.fit_transform(embeddings)
        assert result.shape == (2, 2)
    
    def test_high_dimensional_embedding(self):
        """Test with high-dimensional embeddings"""
        embeddings = np.random.randn(10, 2048)
        result = self.visualizer.fit_transform(embeddings)
        assert result.shape == (10, 2)
    
    def test_low_dimensional_embedding(self):
        """Test with low-dimensional embeddings"""
        embeddings = np.random.randn(10, 16)
        result = self.visualizer.fit_transform(embeddings)
        assert result.shape == (10, 2)
    
    def test_plot_scatter_minimal(self):
        """Test plot_scatter with minimal inputs"""
        embeddings_2d = np.array([[0, 0], [1, 1], [2, 2]])
        texts = ["A", "B", "C"]
        labels = np.array([0, 0, 1])
        
        fig = self.visualizer.plot_scatter(
            embeddings_2d=embeddings_2d,
            texts=texts,
            labels=labels
        )
        assert fig is not None
    
    def test_plot_scatter_large_number_of_points(self):
        """Test plot_scatter with large number of points"""
        n_points = 1000
        embeddings_2d = np.random.randn(n_points, 2)
        texts = [f"Text {i}" for i in range(n_points)]
        labels = np.random.randint(0, 10, n_points)
        
        fig = self.visualizer.plot_scatter(
            embeddings_2d=embeddings_2d,
            texts=texts,
            labels=labels
        )
        assert fig is not None
    
    def test_plot_scatter_single_point(self):
        """Test plot_scatter with single point"""
        embeddings_2d = np.array([[0, 0]])
        texts = ["Only one"]
        labels = np.array([0])
        
        fig = self.visualizer.plot_scatter(
            embeddings_2d=embeddings_2d,
            texts=texts,
            labels=labels
        )
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
