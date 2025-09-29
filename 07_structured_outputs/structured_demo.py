import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:
    from langchain_community.llms import Ollama


# Define structured output schemas
class PersonInfo(BaseModel):
    """Information about a person."""
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years", ge=0, le=150)
    occupation: str = Field(description="Job or profession")
    location: str = Field(description="City and country")
    interests: List[str] = Field(description="List of interests or hobbies", min_items=1, max_items=5)


class ProductReview(BaseModel):
    """Product review with structured data."""
    product_name: str = Field(description="Name of the product")
    rating: int = Field(description="Rating from 1 to 5", ge=1, le=5)
    pros: List[str] = Field(description="List of positive aspects", min_items=1, max_items=3)
    cons: List[str] = Field(description="List of negative aspects", min_items=1, max_items=3)
    recommendation: bool = Field(description="Would recommend this product")
    summary: str = Field(description="Brief summary of the review", max_length=200)


class MeetingNotes(BaseModel):
    """Structured meeting notes."""
    meeting_title: str = Field(description="Title of the meeting")
    date: str = Field(description="Date of the meeting in YYYY-MM-DD format")
    attendees: List[str] = Field(description="List of attendees", min_items=1)
    key_points: List[str] = Field(description="Main discussion points", min_items=2, max_items=5)
    action_items: List[str] = Field(description="Action items with owners", min_items=1, max_items=4)
    next_meeting: Optional[str] = Field(description="Next meeting date if scheduled", default=None)


def create_structured_chain(model_name: str, output_schema: BaseModel):
    """Create a chain that enforces structured output."""
    llm = Ollama(model=model_name)
    parser = PydanticOutputParser(pydantic_object=output_schema)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that extracts structured information from text.

{format_instructions}

Be precise and follow the schema exactly."""),
        ("human", "Extract information from: {input_text}")
    ])
    
    # Create chain with parser
    chain = prompt | llm | parser
    
    return chain, parser


def test_person_extraction(chain, parser):
    """Test person information extraction."""
    test_texts = [
        "Hi, I'm Sarah Johnson, a 28-year-old software engineer from San Francisco, CA. I love hiking, photography, and cooking.",
        "Meet Alex Chen, 35, who works as a data scientist in New York. He enjoys reading, playing guitar, and traveling.",
        "This is Maria Rodriguez, a 42-year-old teacher from Madrid, Spain. She's passionate about art, music, and gardening."
    ]
    
    results = []
    for text in test_texts:
        try:
            result = chain.invoke({"input_text": text})
            results.append({
                "input": text,
                "output": result,
                "success": True
            })
        except Exception as e:
            results.append({
                "input": text,
                "output": None,
                "error": str(e),
                "success": False
            })
    
    return results


def test_product_review(chain, parser):
    """Test product review extraction."""
    test_texts = [
        "I bought the iPhone 15 and it's amazing! The camera quality is incredible and the battery lasts all day. However, it's quite expensive and the storage fills up quickly. I'd definitely recommend it to others. Overall, it's a great phone with some minor drawbacks.",
        "The MacBook Pro is fantastic for work. The performance is excellent and the display is beautiful. But it's very heavy and the price is high. I would recommend it for professionals. Great laptop with premium features.",
        "The AirPods Pro are good but not perfect. The noise cancellation works well and they're comfortable. However, the battery life could be better and they're expensive. I'd recommend them with reservations. Decent earbuds with room for improvement."
    ]
    
    results = []
    for text in test_texts:
        try:
            result = chain.invoke({"input_text": text})
            results.append({
                "input": text,
                "output": result,
                "success": True
            })
        except Exception as e:
            results.append({
                "input": text,
                "output": None,
                "error": str(e),
                "success": False
            })
    
    return results


def test_meeting_notes(chain, parser):
    """Test meeting notes extraction."""
    test_texts = [
        """Weekly team standup on 2024-01-15
Attendees: John, Sarah, Mike, Lisa
We discussed the new feature requirements, reviewed the sprint progress, and planned the next release. 
Action items: John to complete API documentation by Friday, Sarah to review the UI mockups, Mike to set up the testing environment.
Next meeting scheduled for January 22nd.""",
        
        """Project kickoff meeting - 2024-01-10
Team members: Alice, Bob, Charlie, Diana
Main topics: Project scope, timeline, and resource allocation. We also discussed the technical architecture and risk assessment.
Tasks: Alice to create project timeline, Bob to research third-party integrations, Charlie to set up development environment, Diana to prepare stakeholder presentations.
No next meeting scheduled yet."""
    ]
    
    results = []
    for text in test_texts:
        try:
            result = chain.invoke({"input_text": text})
            results.append({
                "input": text,
                "output": result,
                "success": True
            })
        except Exception as e:
            results.append({
                "input": text,
                "output": None,
                "error": str(e),
                "success": False
            })
    
    return results


def save_results_to_markdown(results: dict, output_path: Path):
    """Save structured output results to markdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# Structured Outputs Demo Results

**Time:** {timestamp}  
**Model:** {os.environ.get('OLLAMA_MODEL', 'gemma3:12b-it-qat')}  
**Parser:** PydanticOutputParser

## Overview

This demo shows how to enforce structured outputs using Pydantic schemas and output parsers.

## Person Information Extraction

"""
    
    for i, result in enumerate(results['person'], 1):
        content += f"""### Test {i}
**Input:** {result['input']}

**Output:**
```json
{json.dumps(result['output'].dict() if result['success'] else result.get('error', 'Failed'), indent=2)}
```

**Success:** {result['success']}

---

"""
    
    content += """## Product Review Extraction

"""
    
    for i, result in enumerate(results['review'], 1):
        content += f"""### Test {i}
**Input:** {result['input']}

**Output:**
```json
{json.dumps(result['output'].dict() if result['success'] else result.get('error', 'Failed'), indent=2)}
```

**Success:** {result['success']}

---

"""
    
    content += """## Meeting Notes Extraction

"""
    
    for i, result in enumerate(results['meeting'], 1):
        content += f"""### Test {i}
**Input:** {result['input']}

**Output:**
```json
{json.dumps(result['output'].dict() if result['success'] else result.get('error', 'Failed'), indent=2)}
```

**Success:** {result['success']}

---

"""
    
    content += """## Key Benefits

- **Type Safety**: Pydantic validates data types and constraints
- **Schema Enforcement**: LLM output must match predefined structure
- **Error Handling**: Clear validation errors when schema is violated
- **Integration Ready**: Structured data can be directly used in applications
- **Documentation**: Schema serves as API documentation

## Production Use Cases

- **Data Extraction**: Extract structured data from unstructured text
- **API Integration**: Generate JSON responses for APIs
- **Database Operations**: Create records with validated fields
- **Workflow Automation**: Process documents into structured formats

"""
    
    output_path.write_text(content, encoding="utf-8")


def main():
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:12b-it-qat")
    
    print("=== Structured Outputs Demo ===")
    
    # Test 1: Person Information Extraction
    print("\n1. Testing Person Information Extraction...")
    person_chain, person_parser = create_structured_chain(model_name, PersonInfo)
    person_results = test_person_extraction(person_chain, person_parser)
    
    # Test 2: Product Review Extraction
    print("\n2. Testing Product Review Extraction...")
    review_chain, review_parser = create_structured_chain(model_name, ProductReview)
    review_results = test_product_review(review_chain, review_parser)
    
    # Test 3: Meeting Notes Extraction
    print("\n3. Testing Meeting Notes Extraction...")
    meeting_chain, meeting_parser = create_structured_chain(model_name, MeetingNotes)
    meeting_results = test_meeting_notes(meeting_chain, meeting_parser)
    
    # Compile results
    all_results = {
        'person': person_results,
        'review': review_results,
        'meeting': meeting_results
    }
    
    # Save results
    output_path = Path(__file__).parent / "structured_outputs_results.md"
    save_results_to_markdown(all_results, output_path)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    total_tests = len(person_results) + len(review_results) + len(meeting_results)
    successful_tests = sum(1 for result in person_results + review_results + meeting_results if result['success'])
    
    print(f"\n=== Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")


if __name__ == "__main__":
    main()
