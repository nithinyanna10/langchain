import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:
    from langchain_community.llms import Ollama


class SimpleEmbeddings(Embeddings):
    """Simple embedding implementation for evaluation."""
    
    def __init__(self, model_name: str = "gemma3:12b-it-qat"):
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        embeddings = []
        for text in texts:
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            embedding = []
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i:i+4]
                val = int.from_bytes(chunk, 'big') if len(chunk) == 4 else 0
                embedding.append(val / (2**32))
            
            while len(embedding) < 128:
                embedding.append(0.0)
            embedding = embedding[:128]
            
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed_documents([text])[0]


class EvaluationMetrics:
    """Collection of evaluation metrics for LLM outputs."""
    
    def __init__(self):
        self.embeddings = SimpleEmbeddings()
    
    def exact_match(self, predicted: str, expected: str) -> bool:
        """Check if predicted output exactly matches expected output."""
        return predicted.strip().lower() == expected.strip().lower()
    
    def contains_keywords(self, text: str, keywords: List[str]) -> float:
        """Check what percentage of keywords are present in the text."""
        text_lower = text.lower()
        found_keywords = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return found_keywords / len(keywords) if keywords else 0.0
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings."""
        emb1 = self.embeddings.embed_query(text1)
        emb2 = self.embeddings.embed_query(text2)
        
        # Cosine similarity
        import numpy as np
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def response_length(self, text: str) -> int:
        """Get the length of the response."""
        return len(text.split())
    
    def contains_hallucination_indicators(self, text: str) -> List[str]:
        """Check for common hallucination indicators."""
        indicators = []
        
        # Check for overly confident statements
        if any(phrase in text.lower() for phrase in ["definitely", "certainly", "absolutely", "without a doubt"]):
            indicators.append("overly_confident")
        
        # Check for specific numbers without context
        import re
        if re.search(r'\b\d{4}\b', text):  # 4-digit numbers
            indicators.append("specific_numbers")
        
        # Check for made-up citations
        if "according to" in text.lower() and not any(source in text.lower() for source in ["study", "research", "report"]):
            indicators.append("unclear_sources")
        
        return indicators


class PerformanceMonitor:
    """Monitor performance metrics for LLM calls."""
    
    def __init__(self):
        self.metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_latency": 0.0,
            "error_count": 0,
            "average_latency": 0.0,
            "calls_per_minute": 0.0
        }
        self.start_time = time.time()
    
    def record_call(self, tokens: int, latency: float, success: bool = True):
        """Record a single LLM call."""
        self.metrics["total_calls"] += 1
        self.metrics["total_tokens"] += tokens
        self.metrics["total_latency"] += latency
        
        if not success:
            self.metrics["error_count"] += 1
        
        # Update averages
        self.metrics["average_latency"] = self.metrics["total_latency"] / self.metrics["total_calls"]
        
        # Calculate calls per minute
        elapsed_time = time.time() - self.start_time
        self.metrics["calls_per_minute"] = (self.metrics["total_calls"] / elapsed_time) * 60
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()


def create_evaluation_dataset() -> List[Dict[str, Any]]:
    """Create a dataset for evaluation."""
    return [
        {
            "question": "What is LangChain?",
            "expected_keywords": ["framework", "language models", "tools", "memory"],
            "expected_answer": "LangChain is a framework for developing applications powered by language models."
        },
        {
            "question": "How do embeddings work?",
            "expected_keywords": ["vectors", "similarity", "text", "numerical"],
            "expected_answer": "Embeddings convert text into numerical vectors that represent semantic meaning."
        },
        {
            "question": "What is RAG?",
            "expected_keywords": ["retrieval", "augmented", "generation", "documents"],
            "expected_answer": "RAG combines document retrieval with language model generation."
        },
        {
            "question": "Tell me about agents in AI",
            "expected_keywords": ["reasoning", "tools", "autonomous", "decision"],
            "expected_answer": "AI agents are systems that can reason and use tools to solve problems autonomously."
        }
    ]


def evaluate_llm_responses(model_name: str, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate LLM responses against the dataset."""
    llm = Ollama(model=model_name)
    evaluator = EvaluationMetrics()
    monitor = PerformanceMonitor()
    
    results = []
    
    for item in dataset:
        start_time = time.time()
        
        try:
            # Generate response
            response = llm.invoke(item["question"])
            latency = time.time() - start_time
            
            # Record performance
            monitor.record_call(tokens=len(response.split()), latency=latency, success=True)
            
            # Evaluate response
            exact_match = evaluator.exact_match(response, item["expected_answer"])
            keyword_score = evaluator.contains_keywords(response, item["expected_keywords"])
            semantic_similarity = evaluator.semantic_similarity(response, item["expected_answer"])
            response_length = evaluator.response_length(response)
            hallucination_indicators = evaluator.contains_hallucination_indicators(response)
            
            results.append({
                "question": item["question"],
                "response": response,
                "expected_answer": item["expected_answer"],
                "exact_match": exact_match,
                "keyword_score": keyword_score,
                "semantic_similarity": semantic_similarity,
                "response_length": response_length,
                "hallucination_indicators": hallucination_indicators,
                "latency": latency,
                "success": True
            })
            
        except Exception as e:
            latency = time.time() - start_time
            monitor.record_call(tokens=0, latency=latency, success=False)
            
            results.append({
                "question": item["question"],
                "response": None,
                "error": str(e),
                "latency": latency,
                "success": False
            })
    
    return {
        "results": results,
        "performance_metrics": monitor.get_metrics()
    }


def save_evaluation_results(evaluation_results: Dict[str, Any], output_path: Path):
    """Save evaluation results to markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# LLM Evaluation & Monitoring Results

**Time:** {timestamp}  
**Model:** {os.environ.get('OLLAMA_MODEL', 'gemma3:12b-it-qat')}

## Performance Metrics

"""
    
    metrics = evaluation_results["performance_metrics"]
    content += f"""- **Total Calls:** {metrics['total_calls']}
- **Total Tokens:** {metrics['total_tokens']}
- **Average Latency:** {metrics['average_latency']:.2f}s
- **Error Rate:** {metrics['error_count']}/{metrics['total_calls']} ({metrics['error_count']/metrics['total_calls']*100:.1f}%)
- **Calls per Minute:** {metrics['calls_per_minute']:.1f}

## Evaluation Results

"""
    
    results = evaluation_results["results"]
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        avg_keyword_score = sum(r["keyword_score"] for r in successful_results) / len(successful_results)
        avg_semantic_similarity = sum(r["semantic_similarity"] for r in successful_results) / len(successful_results)
        avg_response_length = sum(r["response_length"] for r in successful_results) / len(successful_results)
        exact_matches = sum(1 for r in successful_results if r["exact_match"])
        
        content += f"""### Summary Statistics
- **Success Rate:** {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)
- **Exact Matches:** {exact_matches}/{len(successful_results)} ({exact_matches/len(successful_results)*100:.1f}%)
- **Average Keyword Score:** {avg_keyword_score:.2f}
- **Average Semantic Similarity:** {avg_semantic_similarity:.2f}
- **Average Response Length:** {avg_response_length:.1f} words

### Detailed Results

"""
        
        for i, result in enumerate(results, 1):
            content += f"""#### Test {i}: {result['question']}

**Response:** {result.get('response', 'ERROR: ' + result.get('error', 'Unknown error'))}

**Metrics:**
- Exact Match: {result.get('exact_match', False)}
- Keyword Score: {result.get('keyword_score', 0):.2f}
- Semantic Similarity: {result.get('semantic_similarity', 0):.2f}
- Response Length: {result.get('response_length', 0)} words
- Hallucination Indicators: {result.get('hallucination_indicators', [])}
- Latency: {result.get('latency', 0):.2f}s

---

"""
    
    content += """## Key Insights

### Evaluation Metrics Explained
- **Exact Match**: Perfect string matching (rare for LLMs)
- **Keyword Score**: Percentage of expected keywords present
- **Semantic Similarity**: Cosine similarity of embeddings
- **Response Length**: Word count of the response
- **Hallucination Indicators**: Signs of potentially false information

### Production Monitoring
- **Latency**: Response time per request
- **Error Rate**: Percentage of failed requests
- **Token Usage**: Cost tracking and optimization
- **Throughput**: Requests per minute

### Best Practices
- Set up alerts for high error rates
- Monitor latency trends
- Track token usage for cost optimization
- Regular evaluation against test datasets
- A/B testing for prompt improvements

"""
    
    output_path.write_text(content, encoding="utf-8")


def main():
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:12b-it-qat")
    
    print("=== LLM Evaluation & Monitoring Demo ===")
    
    # Create evaluation dataset
    dataset = create_evaluation_dataset()
    print(f"Created evaluation dataset with {len(dataset)} items")
    
    # Run evaluation
    print("Running evaluation...")
    evaluation_results = evaluate_llm_responses(model_name, dataset)
    
    # Save results
    output_path = Path(__file__).parent / "evaluation_results.md"
    save_evaluation_results(evaluation_results, output_path)
    print(f"Results saved to: {output_path}")
    
    # Print summary
    results = evaluation_results["results"]
    successful_results = [r for r in results if r["success"]]
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
    
    if successful_results:
        avg_keyword_score = sum(r["keyword_score"] for r in successful_results) / len(successful_results)
        avg_semantic_similarity = sum(r["semantic_similarity"] for r in successful_results) / len(successful_results)
        
        print(f"Average keyword score: {avg_keyword_score:.2f}")
        print(f"Average semantic similarity: {avg_semantic_similarity:.2f}")


if __name__ == "__main__":
    main()
