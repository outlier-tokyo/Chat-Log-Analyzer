import pandas as pd
from .base_loader import BaseLoader

class CSVLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        # TODO: Implement CSV loading logic with error handling
        return pd.read_csv(self.file_path)