import os
from datetime import datetime
from pathlib import Path

from langchain_core.memory import BaseMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)
try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:
    from langchain_community.llms import Ollama


def create_chain_with_memory(memory: BaseMemory, model_name: str = "gemma3:12b-it-qat"):
    llm = Ollama(model=model_name)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Nova's helpful assistant. Keep responses short and friendly."),
        ("human", "{input}"),
    ])
    
    # Load memory variables into the prompt
    chain = (
        RunnablePassthrough.assign(history=memory.load_memory_variables)
        | prompt
        | llm
    )
    
    return chain, memory


def run_conversation(chain, memory, conversation_steps):
    """Run a conversation and return the full history."""
    responses = []
    
    for step in conversation_steps:
        human_input = step["human"]
        response = chain.invoke({"input": human_input})
        
        # Save to memory
        memory.save_context({"input": human_input}, {"output": response})
        
        responses.append({
            "step": step["step"],
            "human": human_input,
            "assistant": response,
            "memory_vars": memory.load_memory_variables({})
        })
    
    return responses


def save_results_to_markdown(results, memory_type, output_path):
    """Save conversation results to markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# Memory Demo: {memory_type}

**Time:** {timestamp}  
**Model:** {os.environ.get('OLLAMA_MODEL', 'gemma3:12b-it-qat')}

## Conversation Flow

"""
    
    for result in results:
        content += f"""### Step {result['step']}

**Human:** {result['human']}

**Assistant:** {result['assistant']}

**Memory Variables:**
```json
{result['memory_vars']}
```

---

"""
    
    output_path.write_text(content, encoding="utf-8")


def main():
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:12b-it-qat")
    
    # Conversation steps to test memory
    conversation_steps = [
        {"step": 1, "human": "Hi! My name is Nova. I'm learning about LangChain memory."},
        {"step": 2, "human": "What's my name?"},
        {"step": 3, "human": "I have a pet cat named Whiskers."},
        {"step": 4, "human": "What's my pet's name?"},
        {"step": 5, "human": "What's my name and my pet's name?"},
    ]
    
    # Test different memory types
    memory_types = {
        "Buffer": ConversationBufferMemory(return_messages=True),
        "BufferWindow": ConversationBufferWindowMemory(k=2, return_messages=True),
        "Summary": ConversationSummaryMemory(
            llm=Ollama(model=model_name),
            return_messages=True
        ),
    }
    
    results = {}
    
    for memory_name, memory in memory_types.items():
        print(f"\n=== Testing {memory_name} Memory ===")
        
        chain, memory = create_chain_with_memory(memory, model_name)
        conversation_results = run_conversation(chain, memory, conversation_steps)
        
        # Save results
        output_path = Path(__file__).parent / f"{memory_name.lower()}_results.md"
        save_results_to_markdown(conversation_results, memory_name, output_path)
        
        results[memory_name] = conversation_results
        
        print(f"Saved results to: {output_path}")
        
        # Show final memory state
        final_memory = memory.load_memory_variables({})
        print(f"Final memory variables: {final_memory}")
    
    # Create comparison summary
    comparison_path = Path(__file__).parent / "memory_comparison.md"
    create_comparison_summary(results, comparison_path)
    print(f"\nComparison saved to: {comparison_path}")


def create_comparison_summary(results, output_path):
    """Create a comparison of all memory types."""
    content = """# Memory Types Comparison

## Key Differences

- **Buffer**: Stores ALL conversation history
- **BufferWindow**: Keeps only last k exchanges (k=2 in demo)
- **Summary**: Uses LLM to compress old context into summaries

## Final Memory States

"""
    
    for memory_type, conversation_results in results.items():
        content += f"### {memory_type}\n\n"
        final_memory = conversation_results[-1]["memory_vars"]
        content += f"```json\n{final_memory}\n```\n\n"
    
    output_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
