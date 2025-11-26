"""Tests for embedding generation."""
import pytest
from src.embeddings.generator import EmbeddingGenerator


def test_embedding_generation():
    """Test embedding generation."""
    generator = EmbeddingGenerator()
    text = "Software Engineer with Python experience"
    embedding = generator.generate_embedding(text)
    
    assert embedding is not None
    assert len(embedding) > 0
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)


def test_batch_embedding_generation():
    """Test batch embedding generation."""
    generator = EmbeddingGenerator()
    texts = [
        "Software Engineer",
        "Data Scientist",
        "Product Manager"
    ]
    embeddings = generator.generate_embeddings_batch(texts)
    
    assert len(embeddings) == len(texts)
    assert all(len(emb) > 0 for emb in embeddings)

