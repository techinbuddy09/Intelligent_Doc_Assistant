
"""
Embedding generation service for the Intelligent Document Assistant.

Purpose:
    Convert document chunks and user queries into dense numerical vectors
    that can be compared using semantic similarity.

Design decisions:
    1. Model: BAAI/bge-small-en-v1.5
       - A lightweight English embedding model suitable for an initial
         local RAG implementation.
       - Produces 384-dimensional vectors.
       - Lower storage and computation requirements than larger models.
       - A starting baseline, not a guarantee of the best retrieval quality.

    2. Device: Automatically select CUDA or CPU.
       - CUDA can accelerate embedding inference on supported NVIDIA GPUs.
       - CPU fallback keeps the service usable without a GPU.

    3. Batch size: 32.
       - Processes multiple chunks together to reduce inference overhead.
       - A reasonable initial value, but actual memory usage depends on
         sequence lengths, model architecture, and available GPU memory.

    4. Normalization: Enabled.
       - Scales each nonzero embedding to unit length.
       - Makes dot product equivalent to cosine similarity for normalized
         vectors, simplifying similarity comparisons.

    5. Query prefix:
       - BGE v1.5 recommends an instruction prefix for short queries
         when performing passage retrieval.
       - Document chunks are embedded without this query instruction.
"""

from functools import lru_cache

import torch
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.

    Why use lru_cache(maxsize=1)?
        Loading a transformer model is relatively expensive.
        Caching avoids loading another copy every time an embedding
        function is called.

        maxsize=1 is sufficient because this service currently uses
        one embedding model.

    Why select the device dynamically?
        The project can use GPU acceleration when CUDA is available
        while retaining compatibility with CPU-only environments.

    Returns:
        A reusable SentenceTransformer instance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer(
        MODEL_NAME,
        device=device,
    )


def embed_documents(documents: list[Document]) -> list[list[float]]:
    """
    Generate embeddings for document chunks.

    Args:
        documents:
            LangChain Document objects produced by the chunking service.

    Why use batch_size=32?
        Batching reduces the overhead of processing chunks individually.
        The value 32 is an initial performance/memory tradeoff, not
        an experimentally optimized setting.

    Why normalize embeddings?
        Unit-length vectors allow cosine similarity to be calculated
        through a dot product, provided both vectors are normalized.

    Why convert_to_numpy=True?
        Returns NumPy arrays, which are convenient for converting into
        plain Python lists and passing to vector database clients.

    Returns:
        One 384-dimensional embedding per document, in input order.

    Important:
        This function generates embeddings only.
        It does not save them to a vector database.
    """
    if not documents:
        return []

    model = get_embedding_model()

    texts = [document.page_content for document in documents]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Generate an embedding for a user's search query.

    Args:
        query:
            The natural-language question to search against the
            stored document chunks.

    Why use the same model as document embedding?
        Query and document vectors must belong to a compatible
        embedding space for meaningful similarity comparisons.

    Why prepend QUERY_PREFIX?
        The BGE model recommends this instruction for short queries
        in passage retrieval tasks. It helps distinguish the search
        query from the passages being searched.

    Why normalize the query embedding?
        The query must use the same normalization convention as
        the stored document embeddings.

    Returns:
        A single 384-dimensional embedding.

    Raises:
        ValueError:
            If the query is empty or contains only whitespace.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    model = get_embedding_model()

    embedding = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embedding.flatten().tolist()
