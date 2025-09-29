import os
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import VectorStoreRetriever
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
            page_content="LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, memory, and tool integration. The framework standardizes common patterns like prompts, chains, memory, and tools.",
            metadata={"source": "langchain_intro", "topic": "framework"}
        ),
        Document(
            page_content="Ollama allows you to run large language models locally on your machine. It supports models like Llama, Mistral, and Gemma. This enables privacy and cost-effective experimentation without cloud APIs.",
            metadata={"source": "ollama_guide", "topic": "local_models"}
        ),
        Document(
            page_content="Vector stores like FAISS, Pinecone, and Chroma are specialized databases for storing and searching embeddings efficiently. They use approximate nearest neighbor algorithms for fast similarity search.",
            metadata={"source": "vectorstore_guide", "topic": "databases"}
        ),
        Document(
            page_content="RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation to provide grounded answers. It retrieves relevant documents and uses them as context for the LLM.",
            metadata={"source": "rag_explanation", "topic": "architecture"}
        ),
        Document(
            page_content="Embeddings convert text into high-dimensional vectors where similar texts have similar vector representations. They enable semantic search and similarity matching.",
            metadata={"source": "embeddings_guide", "topic": "vectors"}
        ),
        Document(
            page_content="Memory in LangChain helps maintain conversation context across multiple interactions. Types include Buffer, Window, and Summary memory for different use cases.",
            metadata={"source": "memory_guide", "topic": "conversation"}
        ),
        Document(
            page_content="Agents use tools and reasoning to solve complex problems by breaking them down into actionable steps. The ReAct pattern combines reasoning and action for dynamic problem solving.",
            metadata={"source": "agents_guide", "topic": "reasoning"}
        ),
        Document(
            page_content="Prompt templates allow you to create reusable prompts with variables for different use cases. They support few-shot learning and structured output generation.",
            metadata={"source": "prompts_guide", "topic": "templates"}
        ),
    ]
    return documents


def build_rag_chain(model_name: str = "gemma3:12b-it-qat"):
    """Build a RAG chain with retrieval and generation."""
    # Create embeddings
    embeddings = SimpleEmbeddings(model_name)
    
    # Create documents
    documents = create_sample_documents()
    
    # Build vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Create retriever
    retriever = VectorStoreRetriever(vectorstore=vectorstore, k=3)
    
    # Create LLM
    llm = Ollama(model=model_name)
    
    # Create RAG prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Nova's helpful assistant. Use the provided context to answer questions accurately and concisely.

Context:
{context}

Instructions:
- Answer based on the provided context
- If the context doesn't contain enough information, say so
- Keep responses clear and helpful
- Cite relevant sources when possible"""),
        ("human", "Question: {question}")
    ])
    
    # Create RAG chain
    def format_docs(docs):
        return "\n\n".join([f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}" for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": lambda x: x["question"]}
        | prompt
        | llm
    )
    
    return rag_chain, vectorstore, retriever


def test_rag_queries(rag_chain, queries: List[str]):
    """Test RAG with different queries."""
    print("\n=== RAG Query Tests ===")
    
    results = []
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        # Get RAG response
        response = rag_chain.invoke({"question": query})
        print(f"RAG Response: {response}")
        
        results.append({
            "query": query,
            "response": response
        })
    
    return results


def save_rag_results(results: List[dict], output_path: Path):
    """Save RAG results to markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# RAG (Retrieval-Augmented Generation) Demo Results

**Time:** {timestamp}  
**Model:** {os.environ.get('OLLAMA_MODEL', 'gemma3:12b-it-qat')}  
**Vector Store:** FAISS  
**Retrieval:** Top 3 similar documents

## RAG Process

1. **Query**: User asks a question
2. **Retrieve**: Find most relevant documents using embeddings
3. **Augment**: Add retrieved context to the prompt
4. **Generate**: LLM answers using the context

## Results

"""
    
    for i, result in enumerate(results, 1):
        content += f"""### Query {i}: "{result['query']}"

**RAG Response:**
{result['response']}

---

"""
    
    content += """## Key Benefits of RAG

- **Grounded Answers**: Responses are based on retrieved documents
- **Up-to-date Information**: Can use recent documents as context
- **Source Attribution**: Can cite where information comes from
- **Reduced Hallucination**: Less likely to make up facts

## Production Use Cases

- **Customer Support**: Answer questions using knowledge base
- **Document Q&A**: Ask questions about specific documents
- **Research Assistant**: Find and summarize relevant information
- **Code Documentation**: Answer questions about codebases

"""
    
    output_path.write_text(content, encoding="utf-8")


def main():
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:12b-it-qat")
    
    print("Building RAG chain...")
    rag_chain, vectorstore, retriever = build_rag_chain(model_name)
    print("RAG chain built successfully!")
    
    # Test queries
    test_queries = [
        "What is LangChain and what does it provide?",
        "How can I run language models locally?",
        "What is RAG and how does it work?",
        "Tell me about different types of memory in LangChain",
        "What are embeddings and why are they useful?",
        "How do agents work and what is the ReAct pattern?",
        "What are the benefits of using prompt templates?",
    ]
    
    # Test RAG
    results = test_rag_queries(rag_chain, test_queries)
    
    # Save results
    output_path = Path(__file__).parent / "rag_results.md"
    save_rag_results(results, output_path)
    print(f"\nRAG results saved to: {output_path}")
    
    # Save vector store for later use
    vectorstore_path = Path(__file__).parent / "rag_vectorstore"
    vectorstore.save_local(str(vectorstore_path))
    print(f"Vector store saved to: {vectorstore_path}")


if __name__ == "__main__":
    main()
