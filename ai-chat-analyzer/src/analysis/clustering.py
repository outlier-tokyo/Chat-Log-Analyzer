from sklearn.cluster import KMeans
import pandas as pd

class TopicClusterer:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def fit_predict(self, vectors):
        # TODO: Implement clustering logic
        return self.model.fit_predict(vectors)