#!/usr/bin/env python
"""Quick test of EmbeddingVisualizer"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from visualization.visualizer import EmbeddingVisualizer

print("[TEST] Starting visualization tests...")

try:
    print("[1] Initializing EmbeddingVisualizer...")
    viz = EmbeddingVisualizer(n_neighbors=5, min_dist=0.1)
    print("✓ Initialization successful")
    
    print("\n[2] Testing fit_transform with 20 samples x 768 dims...")
    embeddings = np.random.randn(20, 768)
    embeddings_2d = viz.fit_transform(embeddings)
    print(f"✓ Output shape: {embeddings_2d.shape}")
    assert embeddings_2d.shape == (20, 2), f"Expected (20, 2), got {embeddings_2d.shape}"
    
    print("\n[3] Testing plot_scatter...")
    texts = [f"Sample text {i}" for i in range(20)]
    labels = np.array([0, 1, 0, 1, 2] * 4)
    
    fig = viz.plot_scatter(
        embeddings_2d=embeddings_2d,
        texts=texts,
        labels=labels,
        title="Test Plot"
    )
    print(f"✓ Figure created: {type(fig).__name__}")
    
    print("\n[4] Testing plot_scatter with noise points...")
    labels_noise = np.array([0, 1, -1, 0, 1, -1, 2] * 2 + [0, 1])
    embeddings_2d_noise = np.random.randn(len(labels_noise), 2)
    texts_noise = [f"Text {i}" for i in range(len(labels_noise))]
    
    fig2 = viz.plot_scatter(
        embeddings_2d=embeddings_2d_noise,
        texts=texts_noise,
        labels=labels_noise,
        title="Noise Test"
    )
    print(f"✓ Noise plot created successfully")
    
    print("\n[5] Testing HTML export...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.html"
        viz.save_html(fig, str(filepath))
        assert filepath.exists(), "HTML file not created"
        print(f"✓ HTML file saved: {filepath.stat().st_size} bytes")
    
    print("\n[6] Testing edge case: single point...")
    embeddings_single = np.array([[0, 0]])
    texts_single = ["Only one"]
    labels_single = np.array([0])
    
    fig3 = viz.plot_scatter(
        embeddings_2d=embeddings_single,
        texts=texts_single,
        labels=labels_single,
        title="Single Point"
    )
    print(f"✓ Single point plot created")
    
    print("\n[7] Testing edge case: large number of points (1000)...")
    embeddings_large = np.random.randn(1000, 2)
    texts_large = [f"Text {i}" for i in range(1000)]
    labels_large = np.random.randint(0, 10, 1000)
    
    fig4 = viz.plot_scatter(
        embeddings_2d=embeddings_large,
        texts=texts_large,
        labels=labels_large,
        title="Large Dataset"
    )
    print(f"✓ Large plot with 1000 points created")
    
    print("\n[8] Testing different UMAP input dimensions...")
    for dim in [128, 256, 512, 768, 1024]:
        embeddings_dim = np.random.randn(10, dim)
        result = viz.fit_transform(embeddings_dim)
        assert result.shape == (10, 2), f"Failed for dim {dim}"
    print(f"✓ All dimension tests passed")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)

except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
