
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # or your embeddings provider

# Initialize embeddings (ensure your API key is set if using OpenAI)
embeddings = OpenAIEmbeddings()


def batch_embed_with_chunks(text: str, chunk_size: int, overlap: int) -> dict:
    """Splits text into chunks, embeds them, and returns metadata."""
    
    # 1. Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    # 2. Split text into chunks
    chunks = splitter.split_text(text)
    
    # 3. Generate embeddings in batch
    vectors = embeddings.embed_documents(chunks)
    
    # 4. Get embedding dimension
    embedding_dim = len(vectors[0]) if vectors else 0
    
    # 5. Return metadata
    return {
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dim": embedding_dim,
        "chunks": chunks
    }