# %% setup_project.py
import os
import json
from pathlib import Path

# プロジェクト名
PROJECT_NAME = "ai-chat-analyzer"

# ディレクトリ構成定義
DIRECTORIES = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/loader",
    "src/preprocessor",
    "src/analysis",
    "src/visualization",
]

# requirements.txt の内容
REQUIREMENTS_CONTENT = """pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
japanize-matplotlib>=1.1.3
wordcloud>=1.9.0
mecab-python3>=1.0.6
unidic-lite>=1.0.8
scikit-learn>=1.3.0
sentence-transformers>=2.2.2
transformers>=4.30.0
torch>=2.0.0
networkx>=3.1
openai>=1.0.0
python-dotenv>=1.0.0
tqdm>=4.65.0
jupyterlab>=4.0.0
ipykernel>=6.25.0
datasets>=2.14.0
"""

# ソースコードの雛形定義
FILE_TEMPLATES = {
    # --- Config ---
    "src/config.py": """
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATA_PATH_RAW = "data/raw"
    DATA_PATH_PROCESSED = "data/processed"
    MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens" # Example for Japanese SBERT
""",

    # --- Loaders ---
    "src/loader/base_loader.py": """
from abc import ABC, abstractmethod
import pandas as pd

class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass
""",
    "src/loader/csv_loader.py": """
import pandas as pd
from .base_loader import BaseLoader

class CSVLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        # TODO: Implement CSV loading logic with error handling
        return pd.read_csv(self.file_path)
""",
    "src/loader/huggingface_loader.py": """
import pandas as pd
from datasets import load_dataset
from .base_loader import BaseLoader

class HuggingFaceLoader(BaseLoader):
    def __init__(self, dataset_name="nu-dialogue/real-persona-chat"):
        self.dataset_name = dataset_name

    def load(self) -> pd.DataFrame:
        print(f"Loading dataset: {self.dataset_name}...")
        # dataset = load_dataset(self.dataset_name)
        # TODO: Implement conversion from HF Dataset to Pandas DataFrame
        # TODO: Normalize columns to match the standard schema
        return pd.DataFrame() 
""",

    # --- Preprocessors ---
    "src/preprocessor/text_cleaner.py": """
import re

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        # TODO: Implement text cleaning (remove HTML tags, specific symbols, etc.)
        if not isinstance(text, str):
            return ""
        text = text.strip()
        return text
""",
    "src/preprocessor/tokenizer.py": """
import MeCab

class Tokenizer:
    def __init__(self):
        # unidic-lite is used by default
        self.tagger = MeCab.Tagger()

    def tokenize(self, text: str) -> list[str]:
        # TODO: Implement tokenization logic
        # node = self.tagger.parseToNode(text)
        return []
""",

    # --- Analysis Engines ---
    "src/analysis/clustering.py": """
from sklearn.cluster import KMeans
import pandas as pd

class TopicClusterer:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def fit_predict(self, vectors):
        # TODO: Implement clustering logic
        return self.model.fit_predict(vectors)
""",
    "src/analysis/vectorizer.py": """
from sentence_transformers import SentenceTransformer

class TextVectorizer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        # TODO: Implement vectorization
        return self.model.encode(texts, show_progress_bar=True)
""",
    "src/analysis/cooccurrence.py": """
import networkx as nx
import pandas as pd

class CooccurrenceNetwork:
    def build_network(self, tokenized_docs: list[list[str]]):
        # TODO: Calculate co-occurrence matrix and build NetworkX graph
        G = nx.Graph()
        return G
""",
    "src/analysis/llm_wrapper.py": """
# import openai

class LLMSummarizer:
    def __init__(self, api_key):
        self.api_key = api_key

    def summarize(self, text: str) -> str:
        # TODO: Call OpenAI API (ChatCompletion) for summarization
        return "Summary placeholder"
""",

    # --- Visualization ---
    "src/visualization/charts.py": """
import plotly.express as px
import pandas as pd

def plot_user_demographics(df: pd.DataFrame):
    # TODO: Create visualization for user demographics
    pass

def plot_topic_distribution(df: pd.DataFrame):
    # TODO: Create visualization for topic clusters
    pass
"""
}

def create_notebook(filename):
    """Creates a minimal valid Jupyter Notebook file."""
    nb_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {filename}\n", "\n", "Initialize analysis here."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "%load_ext autoreload\n",
                    "%autoreload 2\n",
                    "import sys\n",
                    "sys.path.append('../src')\n",
                    "from config import Config"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return json.dumps(nb_content, indent=2)

# %% Main Execution
def main():
    base_path = Path(PROJECT_NAME)
    
    # 1. Create Directories
    print(f"Creating project structure for '{PROJECT_NAME}'...")
    if not base_path.exists():
        base_path.mkdir()
    
    for directory in DIRECTORIES:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
        # Add .gitkeep to empty directories
        with open(base_path / directory / ".gitkeep", "w") as f:
            pass

    # 2. Create Files from Templates
    for rel_path, content in FILE_TEMPLATES.items():
        file_path = base_path / rel_path
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"Created: {rel_path}")

    # 3. Create Special Files
    # __init__.py for src
    with open(base_path / "src" / "__init__.py", "w") as f:
        pass
        
    # requirements.txt
    with open(base_path / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS_CONTENT)

    # Notebooks
    notebooks = ["01_overview.ipynb", "02_user_analysis.ipynb", "03_topic_clustering.ipynb"]
    for nb in notebooks:
        with open(base_path / "notebooks" / nb, "w", encoding="utf-8") as f:
            f.write(create_notebook(nb))
            
    print("\nSuccessfully created project structure!")
    print(f"Next steps:\n1. cd {PROJECT_NAME}\n2. python -m venv venv\n3. pip install -r requirements.txt")

# %% Run the script
if __name__ == "__main__":
    main()