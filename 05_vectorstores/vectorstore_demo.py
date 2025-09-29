import os
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:
    from langchain_community.llms import Ollama


class SimpleEmbeddings(Embeddings):
    """Simple embedding implementation for demo purposes."""
    
    def __init__(self, model_name: str = "gemma3:12b-it-qat"):
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        # For demo, create simple hash-based embeddings
        embeddings = []
        for text in texts:
            # Simple hash-based embedding (not production quality)
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Convert to 128-dimensional vector
            embedding = []
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i:i+4]
                val = int.from_bytes(chunk, 'big') if len(chunk) == 4 else 0
                embedding.append(val / (2**32))  # Normalize
            
            # Pad or truncate to 128 dimensions
            while len(embedding) < 128:
                embedding.append(0.0)
            embedding = embedding[:128]
            
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed_documents([text])[0]


def create_sample_documents() -> List[Document]:
    """Create sample documents for the knowledge base."""
    documents = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, memory, and tool integration.",
            metadata={"source": "langchain_intro", "topic": "framework"}
        ),
        Document(
            page_content="Ollama allows you to run large language models locally on your machine. It supports models like Llama, Mistral, and Gemma.",
            metadata={"source": "ollama_guide", "topic": "local_models"}
        ),
        Document(
            page_content="Vector stores like FAISS, Pinecone, and Chroma are specialized databases for storing and searching embeddings efficiently.",
            metadata={"source": "vectorstore_guide", "topic": "databases"}
        ),
        Document(
            page_content="RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation to provide grounded answers.",
            metadata={"source": "rag_explanation", "topic": "architecture"}
        ),
        Document(
            page_content="Embeddings convert text into high-dimensional vectors where similar texts have similar vector representations.",
            metadata={"source": "embeddings_guide", "topic": "vectors"}
        ),
        Document(
            page_content="Memory in LangChain helps maintain conversation context across multiple interactions with the language model.",
            metadata={"source": "memory_guide", "topic": "conversation"}
        ),
        Document(
            page_content="Agents use tools and reasoning to solve complex problems by breaking them down into actionable steps.",
            metadata={"source": "agents_guide", "topic": "reasoning"}
        ),
        Document(
            page_content="Prompt templates allow you to create reusable prompts with variables for different use cases.",
            metadata={"source": "prompts_guide", "topic": "templates"}
        ),
    ]
    return documents


def build_vectorstore(documents: List[Document], embeddings: Embeddings) -> VectorStore:
    """Build a vector store from documents."""
    print("Building vector store...")
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print(f"Vector store created with {len(documents)} documents")
    return vectorstore


def test_similarity_search(vectorstore: VectorStore, queries: List[str]):
    """Test similarity search with different queries."""
    print("\n=== Similarity Search Tests ===")
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Search for similar documents
        results = vectorstore.similarity_search(query, k=3)
        
        print("Top 3 similar documents:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content[:100]}...")
            print(f"   Source: {doc.metadata.get('source', 'unknown')}")
            print(f"   Topic: {doc.metadata.get('topic', 'unknown')}")


def save_search_results(vectorstore: VectorStore, queries: List[str], output_path: Path):
    """Save similarity search results to markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# Vector Store Similarity Search Results

**Time:** {timestamp}  
**Model:** {os.environ.get('OLLAMA_MODEL', 'gemma3:12b-it-qat')}  
**Vector Store:** FAISS  
**Documents:** 8 sample documents

## Search Results

"""
    
    for query in queries:
        content += f"""### Query: "{query}"

"""
        
        results = vectorstore.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            content += f"""**Result {i}:**
- **Content:** {doc.page_content}
- **Source:** {doc.metadata.get('source', 'unknown')}
- **Topic:** {doc.metadata.get('topic', 'unknown')}

"""
        
        content += "---\n\n"
    
    output_path.write_text(content, encoding="utf-8")


def main():
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:12b-it-qat")
    
    # Create embeddings
    embeddings = SimpleEmbeddings(model_name)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")
    
    # Build vector store
    vectorstore = build_vectorstore(documents, embeddings)
    
    # Test queries
    test_queries = [
        "What is LangChain?",
        "How do I run models locally?",
        "What are embeddings?",
        "How does RAG work?",
        "Tell me about memory in conversations",
        "What are agents and how do they work?",
    ]
    
    # Test similarity search
    test_similarity_search(vectorstore, test_queries)
    
    # Save results
    output_path = Path(__file__).parent / "vectorstore_results.md"
    save_search_results(vectorstore, test_queries, output_path)
    print(f"\nResults saved to: {output_path}")
    
    # Save vector store for later use
    vectorstore_path = Path(__file__).parent / "vectorstore"
    vectorstore.save_local(str(vectorstore_path))
    print(f"Vector store saved to: {vectorstore_path}")


if __name__ == "__main__":
    main()
