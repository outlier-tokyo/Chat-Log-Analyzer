from sentence_transformers import SentenceTransformer

class TextVectorizer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        # TODO: Implement vectorization
        return self.model.encode(texts, show_progress_bar=True)