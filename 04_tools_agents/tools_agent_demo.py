import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:
    from langchain_community.llms import Ollama


# Define tools
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely. Input should be a valid Python expression."""
    try:
        # Simple safety check - only allow basic math operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations allowed"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Simulate web search results. In real implementation, this would call a search API."""
    # Mock search results for demo
    mock_results = {
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "ollama": "Ollama is a tool for running large language models locally on your machine.",
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "ai": "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.",
        "weather": "Weather refers to atmospheric conditions including temperature, humidity, and precipitation.",
    }
    
    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return f"Search results for '{query}': {value}"
    
    return f"Search results for '{query}': No specific results found, but this is a simulated search."


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


class ReActAgent:
    """Simple ReAct agent implementation."""
    
    def __init__(self, model_name: str = "gemma3:12b-it-qat"):
        self.llm = Ollama(model=model_name)
        self.tools = [calculator, web_search, get_current_time]
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        self.system_prompt = """You are Nova's helpful assistant. You can use tools to solve problems.

Available tools:
- calculator(expression): Calculate mathematical expressions
- web_search(query): Search for information (simulated)
- get_current_time(): Get current date and time

Use this format:
Thought: [Your reasoning about what to do next]
Action: [Tool name]
Action Input: [Input for the tool]

When you have the final answer, respond with:
Final Answer: [Your complete response]

Be concise and helpful."""

    def parse_llm_output(self, output: str) -> Dict[str, str]:
        """Parse LLM output to extract thought, action, and action input."""
        thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|Final Answer:|$)', output, re.DOTALL)
        action_match = re.search(r'Action:\s*(\w+)', output)
        action_input_match = re.search(r'Action Input:\s*(.*?)(?=Thought:|Final Answer:|$)', output, re.DOTALL)
        final_answer_match = re.search(r'Final Answer:\s*(.*)', output, re.DOTALL)
        
        return {
            'thought': thought_match.group(1).strip() if thought_match else '',
            'action': action_match.group(1) if action_match else '',
            'action_input': action_input_match.group(1).strip() if action_input_match else '',
            'final_answer': final_answer_match.group(1).strip() if final_answer_match else ''
        }

    def run_tool(self, action: str, action_input: str) -> str:
        """Execute a tool with the given input."""
        if action in self.tool_map:
            try:
                return self.tool_map[action].invoke(action_input)
            except Exception as e:
                return f"Tool error: {str(e)}"
        else:
            return f"Unknown tool: {action}"

    def solve(self, problem: str, max_steps: int = 5) -> Dict[str, Any]:
        """Solve a problem using ReAct reasoning."""
        conversation = []
        step = 0
        
        prompt = f"{self.system_prompt}\n\nProblem: {problem}\n\nLet's solve this step by step."
        
        while step < max_steps:
            step += 1
            print(f"\n--- Step {step} ---")
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            print(f"LLM Response: {response}")
            
            # Parse the response
            parsed = self.parse_llm_output(response)
            conversation.append({
                'step': step,
                'llm_response': response,
                'parsed': parsed
            })
            
            # Check if we have a final answer
            if parsed['final_answer']:
                print(f"Final Answer: {parsed['final_answer']}")
                break
            
            # Execute action if present
            if parsed['action'] and parsed['action_input']:
                tool_result = self.run_tool(parsed['action'], parsed['action_input'])
                print(f"Tool Result: {tool_result}")
                
                # Add tool result to conversation
                conversation.append({
                    'step': step,
                    'tool_result': tool_result
                })
                
                # Update prompt with tool result
                prompt += f"\n\nTool Result: {tool_result}\n\nContinue reasoning:"
            else:
                print("No valid action found, stopping.")
                break
        
        return {
            'problem': problem,
            'conversation': conversation,
            'final_answer': parsed.get('final_answer', 'No final answer reached'),
            'steps_taken': step
        }


def save_agent_trace(result: Dict[str, Any], output_path: Path):
    """Save agent reasoning trace to markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# ReAct Agent Trace

**Time:** {timestamp}  
**Model:** {os.environ.get('OLLAMA_MODEL', 'gemma3:12b-it-qat')}  
**Problem:** {result['problem']}  
**Steps Taken:** {result['steps_taken']}

## Final Answer
{result['final_answer']}

## Reasoning Trace

"""
    
    for item in result['conversation']:
        if 'llm_response' in item:
            content += f"""### Step {item['step']}: LLM Reasoning

**Response:**
```
{item['llm_response']}
```

**Parsed:**
- Thought: {item['parsed']['thought']}
- Action: {item['parsed']['action']}
- Action Input: {item['parsed']['action_input']}
- Final Answer: {item['parsed']['final_answer']}

"""
        elif 'tool_result' in item:
            content += f"""### Step {item['step']}: Tool Execution

**Result:**
```
{item['tool_result']}
```

"""
    
    output_path.write_text(content, encoding="utf-8")


def main():
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:12b-it-qat")
    agent = ReActAgent(model_name)
    
    # Test problems
    problems = [
        "What is 15 * 23 + 45?",
        "What is the current time?",
        "Tell me about LangChain and also calculate 2^8",
        "What's the weather like today? (simulated search)",
    ]
    
    results = []
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*50}")
        print(f"PROBLEM {i}: {problem}")
        print(f"{'='*50}")
        
        result = agent.solve(problem)
        results.append(result)
        
        # Save individual trace
        output_path = Path(__file__).parent / f"agent_trace_{i}.md"
        save_agent_trace(result, output_path)
        print(f"Saved trace to: {output_path}")
    
    # Create summary
    summary_path = Path(__file__).parent / "agent_summary.md"
    create_summary(results, summary_path)
    print(f"\nSummary saved to: {summary_path}")


def create_summary(results: List[Dict[str, Any]], output_path: Path):
    """Create a summary of all agent runs."""
    content = """# ReAct Agent Summary

## Overview
This demonstrates how a ReAct agent uses tools to solve different types of problems.

## Results Summary

"""
    
    for i, result in enumerate(results, 1):
        content += f"""### Problem {i}
**Question:** {result['problem']}
**Answer:** {result['final_answer']}
**Steps:** {result['steps_taken']}

"""
    
    content += """## Key Insights
- **ReAct Pattern**: The agent reasons about what to do, then acts with tools
- **Tool Selection**: Agent chooses appropriate tools based on problem type
- **Iterative Process**: Agent can use multiple tools in sequence
- **Reasoning Transparency**: Each step shows the agent's thought process

"""
    
    output_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
