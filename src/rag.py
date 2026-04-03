"""
RAG (Retrieval-Augmented Generation) module for the Solar Energy Grid Optimization Assistant.

This module loads knowledge documents from the `knowledge/` directory,
splits them into chunks, embeds them using sentence-transformers,
and stores them for fast similarity search using sklearn NearestNeighbors.
"""

import os
import glob
import json
import hashlib
import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# Paths
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge')
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Embedding model (small, fast, runs locally — no API needed)
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'


def load_documents(knowledge_dir: str = KNOWLEDGE_DIR) -> list[dict]:
    """Load all markdown files from the knowledge directory."""
    documents = []
    for filepath in sorted(glob.glob(os.path.join(knowledge_dir, '*.md'))):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append({
            'source': os.path.basename(filepath),
            'content': content
        })
    return documents


def chunk_documents(documents: list[dict], chunk_size: int = 300) -> list[dict]:
    """
    Split documents into smaller chunks for better retrieval precision.
    Each chunk is roughly `chunk_size` words.
    """
    chunks = []
    for doc in documents:
        lines = doc['content'].split('\n')
        current_chunk = []
        current_word_count = 0

        for line in lines:
            word_count = len(line.split())
            if current_word_count + word_count > chunk_size and current_chunk:
                chunks.append({
                    'source': doc['source'],
                    'text': '\n'.join(current_chunk)
                })
                current_chunk = [line]
                current_word_count = word_count
            else:
                current_chunk.append(line)
                current_word_count += word_count

        # Add the last chunk
        if current_chunk:
            chunks.append({
                'source': doc['source'],
                'text': '\n'.join(current_chunk)
            })
    return chunks


class RAGRetriever:
    """Sklearn-based retrieval system for grid management knowledge."""

    def __init__(self, persist_index: bool | None = None, index_dir: str | None = None):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.nn_index: NearestNeighbors | None = None
        self.index_dir = index_dir or os.getenv('RAG_INDEX_DIR', DEFAULT_MODELS_DIR)
        if persist_index is None:
            persist_env = os.getenv('RAG_PERSIST_INDEX', 'true').strip().lower()
            self.persist_index = persist_env in {'1', 'true', 'yes', 'on'}
        else:
            self.persist_index = persist_index

    @property
    def chunks_path(self) -> str:
        return os.path.join(self.index_dir, 'rag_chunks.json')

    @property
    def embeddings_path(self) -> str:
        return os.path.join(self.index_dir, 'rag_embeddings.npy')

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.index_dir, 'rag_index_manifest.json')

    @staticmethod
    def _sha256_file(filepath: str) -> str:
        """Compute SHA-256 for a file to support integrity checks."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                hasher.update(block)
        return hasher.hexdigest()

    def _save_index(self):
        """Persist chunks/embeddings using safe formats and write an integrity manifest."""
        if self.embeddings is None:
            raise ValueError("Embeddings are not initialized.")

        os.makedirs(self.index_dir, exist_ok=True)

        # Verify we can write to the destination before attempting persistence.
        probe_path = os.path.join(self.index_dir, '.rag_write_test')
        try:
            with open(probe_path, 'w', encoding='utf-8') as probe_file:
                probe_file.write('ok')
            os.remove(probe_path)
        except OSError as exc:
            raise PermissionError(f"Index directory is not writable: {self.index_dir}") from exc

        with open(self.chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        np.save(self.embeddings_path, self.embeddings)

        manifest = {
            'version': 1,
            'embedding_model': EMBED_MODEL_NAME,
            'chunks_path': os.path.basename(self.chunks_path),
            'embeddings_path': os.path.basename(self.embeddings_path),
            'chunks_sha256': self._sha256_file(self.chunks_path),
            'embeddings_sha256': self._sha256_file(self.embeddings_path),
            'num_chunks': len(self.chunks),
            'embeddings_shape': list(self.embeddings.shape),
            'embeddings_dtype': str(self.embeddings.dtype)
        }

        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

    def _load_index_from_disk(self):
        """Load index files from disk after validating integrity and shape consistency."""
        required_paths = [self.chunks_path, self.embeddings_path, self.manifest_path]
        if not all(os.path.exists(path) for path in required_paths):
            raise FileNotFoundError("One or more index files are missing.")

        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        if manifest.get('chunks_sha256') != self._sha256_file(self.chunks_path):
            raise ValueError("Chunk index integrity check failed.")
        if manifest.get('embeddings_sha256') != self._sha256_file(self.embeddings_path):
            raise ValueError("Embeddings index integrity check failed.")

        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        embeddings = np.load(self.embeddings_path, allow_pickle=False)

        if not isinstance(chunks, list):
            raise ValueError("Invalid chunk index format.")
        if embeddings.ndim != 2:
            raise ValueError("Invalid embeddings shape.")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunk and embedding counts do not match.")

        self.chunks = chunks
        self.embeddings = embeddings.astype('float32', copy=False)

    def build_index(self, knowledge_dir: str = KNOWLEDGE_DIR):
        """Load documents, chunk them, embed, and build the search index."""
        documents = load_documents(knowledge_dir)
        if not documents:
            raise FileNotFoundError(f"No .md files found in {knowledge_dir}")

        self.chunks = chunk_documents(documents)
        texts = [chunk['text'] for chunk in self.chunks]

        # Generate embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=False)
        self.embeddings = np.array(self.embeddings, dtype='float32')

        # Build NearestNeighbors index
        self.nn_index = NearestNeighbors(n_neighbors=min(5, len(self.chunks)), metric='cosine')
        self.nn_index.fit(self.embeddings)

        # Persist index unless disabled or unavailable (read-only environments).
        if self.persist_index:
            try:
                self._save_index()
            except OSError as exc:
                warnings.warn(
                    f"RAG index persistence disabled due to write failure: {exc}",
                    RuntimeWarning
                )
                self.persist_index = False

        return len(self.chunks)

    def _load_index(self):
        """Load a previously saved index from disk."""
        if not self.persist_index:
            self.build_index()
            return

        try:
            self._load_index_from_disk()
            self.nn_index = NearestNeighbors(n_neighbors=min(5, len(self.chunks)), metric='cosine')
            self.nn_index.fit(self.embeddings)
        except (OSError, ValueError, json.JSONDecodeError):
            self.build_index()

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a given query.

        Returns a list of dicts with:
        - 'source': markdown filename
        - 'text': retrieved chunk
        - 'distance': cosine distance from `kneighbors` (lower is better, typically 0 to 2)
        - 'similarity': convenience score computed as `1 - distance` (higher is better, typically -1 to 1)
        """
        if self.nn_index is None:
            self._load_index()

        # Encode query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype='float32')

        # Search
        distances, indices = self.nn_index.kneighbors(query_embedding, n_neighbors=min(k, len(self.chunks)))

        results = []
        for i, idx in enumerate(indices[0]):
            distance = float(distances[0][i])
            results.append({
                'source': self.chunks[idx]['source'],
                'text': self.chunks[idx]['text'],
                'distance': distance,
                'similarity': 1.0 - distance
            })
        return results


# Singleton instance for use across the app
_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    """Get or create the singleton RAGRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
        _retriever._load_index()
    return _retriever


if __name__ == '__main__':
    # Quick test
    retriever = get_retriever()
    print(f"Built index with {len(retriever.chunks)} chunks")

    results = retriever.retrieve("How should I handle battery storage during peak hours?")
    for r in results:
        print(
            f"\n--- Source: {r['source']} "
            f"(distance: {r['distance']:.4f}, similarity: {r['similarity']:.4f}) ---"
        )
        print(r['text'][:200])
