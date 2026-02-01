import networkx as nx
import pandas as pd

class CooccurrenceNetwork:
    def build_network(self, tokenized_docs: list[list[str]]):
        # TODO: Calculate co-occurrence matrix and build NetworkX graph
        G = nx.Graph()
        return G