import MeCab

class Tokenizer:
    def __init__(self):
        # unidic-lite is used by default
        self.tagger = MeCab.Tagger()

    def tokenize(self, text: str) -> list[str]:
        # TODO: Implement tokenization logic
        # node = self.tagger.parseToNode(text)
        return []