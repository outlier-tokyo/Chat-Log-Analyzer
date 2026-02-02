import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("=" * 60)
print("VISUALIZER TEST DIAGNOSTIC")
print("=" * 60)

try:
    print("\n[1] Importing dependencies...")
    import numpy as np
    print("  ✓ numpy")
    import plotly.graph_objects as go
    print("  ✓ plotly")
    import umap
    print("  ✓ umap")
    
    print("\n[2] Importing EmbeddingVisualizer...")
    from visualization.visualizer import EmbeddingVisualizer
    print("  ✓ EmbeddingVisualizer imported")
    
    print("\n[3] Creating visualizer instance...")
    viz = EmbeddingVisualizer()
    print("  ✓ Instance created")
    
    print("\n[4] Testing fit_transform...")
    embeddings = np.random.randn(10, 768)
    result = viz.fit_transform(embeddings)
    print(f"  ✓ Output shape: {result.shape}")
    
    print("\n[5] Testing plot_scatter...")
    texts = [f"Text {i}" for i in range(10)]
    labels = np.array([0, 1] * 5)
    fig = viz.plot_scatter(embeddings_2d=result, texts=texts, labels=labels)
    print(f"  ✓ Figure created: {type(fig).__name__}")
    
    print("\n" + "=" * 60)
    print("✓ ALL DIAGNOSTICS PASSED")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ ERROR at step:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
