import os
import pickle
import threading
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv

# new client-style import
from openai import OpenAI
from utils.text_utils import chunk_text


# load environment variables early so key is available
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not configured; set it in .env or environment")
client = OpenAI(api_key=api_key)

# OpenAI embedding model and its output dimension.
# "text-embedding-3-small" currently produces 1536-dimensional vectors.
EMBED_MODEL = "text-embedding-3-small"
DIMENSION = 1536


class EmbeddingService:
    """Handles creating and querying a FAISS vector store of text embeddings.

    The index and the list of texts are persisted to disk so that the service
    can be restarted without losing previously indexed books.  By default the
    files are stored in ``data/index.faiss`` and ``data/texts.pkl``.
    """

    def __init__(self,
                 index_path: str = "data/index.faiss",
                 texts_path: str = "data/texts.pkl") -> None:
        # ensure storage directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.index_path = index_path
        self.texts_path = texts_path
        self._lock = threading.Lock()

        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            # load existing index and texts
            self.index = faiss.read_index(self.index_path)
            with open(self.texts_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            # fresh index
            self.index = faiss.IndexFlatIP(DIMENSION)
            self.texts: List[str] = []

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # using new OpenAI client interface
        response = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def index_text(self, text: str) -> List[int]:
        """Split a document into chunks, embed them and add to the index.

        Returns the list of internal ids that were added.
        """
        chunks = chunk_text(text, min_size=500, max_size=1000)
        if not chunks:
            return []
        embeddings = self._embed(chunks)
        arr = np.array(embeddings, dtype="float32")
        # faiss expects normalized vectors for cosine similarity
        faiss.normalize_L2(arr)
        self.index.add(arr)

        start_id = len(self.texts)
        self.texts.extend(chunks)
        return list(range(start_id, start_id + len(chunks)))

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Return the top_k most similar text chunks for the provided query."""
        if self.index.ntotal == 0:
            return []
        emb = self._embed([query])[0]
        vec = np.array([emb], dtype="float32")
        faiss.normalize_L2(vec)
        distances, indices = self.index.search(vec, top_k)
        result: List[str] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.texts):
                result.append(self.texts[idx])
        return result

    def save(self) -> None:
        """Persist the current index and text list to disk."""
        # locking to avoid concurrent writes
        with self._lock:
            faiss.write_index(self.index, self.index_path)
            with open(self.texts_path, "wb") as f:
                pickle.dump(self.texts, f)
