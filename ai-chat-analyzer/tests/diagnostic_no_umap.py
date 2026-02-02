#!/usr/bin/env python
"""Quick diagnostic for visualizer without UMAP issues"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("=" * 60)
print("VISUALIZER DIAGNOSTIC (NO UMAP)")
print("=" * 60)

try:
    print("\n[1] Importing dependencies...")
    import numpy as np
    print("  ✓ numpy")
    import pandas as pd
    print("  ✓ pandas")
    import plotly.graph_objects as go
    import plotly.express as px
    print("  ✓ plotly")
    
    print("\n[2] Checking for PCA (fallback)...")
    from sklearn.decomposition import PCA
    print("  ✓ PCA available")
    
    print("\n[3] Creating test data...")
    embeddings = np.random.randn(10, 768)
    embeddings_2d_pca = PCA(n_components=2).fit_transform(embeddings)
    print(f"  ✓ PCA 2D shape: {embeddings_2d_pca.shape}")
    
    print("\n[4] Creating scatter plot with Plotly...")
    texts = [f"Text {i}" for i in range(10)]
    labels = np.array([0, 1] * 5)
    
    data = {
        'x': embeddings_2d_pca[:, 0],
        'y': embeddings_2d_pca[:, 1],
        'text': texts,
        'cluster': labels.astype(str)
    }
    
    df = pd.DataFrame(data)
    print(f"  ✓ DataFrame created: {df.shape}")
    
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        hover_data={'text': True},
        title='Test Scatter Plot'
    )
    print(f"  ✓ Figure created: {type(fig).__name__}")
    
    print("\n[5] Checking import of EmbeddingVisualizer...")
    from visualization.visualizer import EmbeddingVisualizer
    print("  ✓ EmbeddingVisualizer imported (with UMAP fallback)")
    
    print("\n" + "=" * 60)
    print("✓ ALL DIAGNOSTICS PASSED")
    print("  - PCA available as UMAP fallback")
    print("  - Plotly scatter plots working")
    print("  - EmbeddingVisualizer ready")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ ERROR:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
