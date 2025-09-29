# Structured Outputs Demo Results

**Time:** 2025-09-28 21:52:40  
**Model:** gemma3:12b-it-qat  
**Parser:** PydanticOutputParser

## Overview

This demo shows how to enforce structured outputs using Pydantic schemas and output parsers.

## Person Information Extraction

### Test 1
**Input:** Hi, I'm Sarah Johnson, a 28-year-old software engineer from San Francisco, CA. I love hiking, photography, and cooking.

**Output:**
```json
"\"Input to ChatPromptTemplate is missing variables {'format_instructions'}.  Expected: ['format_instructions', 'input_text'] Received: ['input_text']\\nNote: if you intended {format_instructions} to be part of the string and not a variable, please escape it with double curly braces like: '{{format_instructions}}'.\\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_PROMPT_INPUT \""
```

**Success:** False

---

### Test 2
**Input:** Meet Alex Chen, 35, who works as a data scientist in New York. He enjoys reading, playing guitar, and traveling.

**Output:**
```json
"\"Input to ChatPromptTemplate is missing variables {'format_instructions'}.  Expected: ['format_instructions', 'input_text'] Received: ['input_text']\\nNote: if you intended {format_instructions} to be part of the string and not a variable, please escape it with double curly braces like: '{{format_instructions}}'.\\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_PROMPT_INPUT \""
```

**Success:** False

---

### Test 3
**Input:** This is Maria Rodriguez, a 42-year-old teacher from Madrid, Spain. She's passionate about art, music, and gardening.

**Output:**
```json
"\"Input to ChatPromptTemplate is missing variables {'format_instructions'}.  Expected: ['format_instructions', 'input_text'] Received: ['input_text']\\nNote: if you intended {format_instructions} to be part of the string and not a variable, please escape it with double curly braces like: '{{format_instructions}}'.\\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_PROMPT_INPUT \""
```

**Success:** False

---

## Product Review Extraction

### Test 1
**Input:** I bought the iPhone 15 and it's amazing! The camera quality is incredible and the battery lasts all day. However, it's quite expensive and the storage fills up quickly. I'd definitely recommend it to others. Overall, it's a great phone with some minor drawbacks.

**Output:**
```json
"\"Input to ChatPromptTemplate is missing variables {'format_instructions'}.  Expected: ['format_instructions', 'input_text'] Received: ['input_text']\\nNote: if you intended {format_instructions} to be part of the string and not a variable, please escape it with double curly braces like: '{{format_instructions}}'.\\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_PROMPT_INPUT \""
```

**Success:** False

---

### Test 2
**Input:** The MacBook Pro is fantastic for work. The performance is excellent and the display is beautiful. But it's very heavy and the price is high. I would recommend it for professionals. Great laptop with premium features.

**Output:**
```json
"\"Input to ChatPromptTemplate is missing variables {'format_instructions'}.  Expected: ['format_instructions', 'input_text'] Received: ['input_text']\\nNote: if you intended {format_instructions} to be part of the string and not a variable, please escape it with double curly braces like: '{{format_instructions}}'.\\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_PROMPT_INPUT \""
```

**Success:** False

---

### Test 3
**Input:** The AirPods Pro are good but not perfect. The noise cancellation works well and they're comfortable. However, the battery life could be better and they're expensive. I'd recommend them with reservations. Decent earbuds with room for improvement.

**Output:**
```json
"\"Input to ChatPromptTemplate is missing variables {'format_instructions'}.  Expected: ['format_instructions', 'input_text'] Received: ['input_text']\\nNote: if you intended {format_instructions} to be part of the string and not a variable, please escape it with double curly braces like: '{{format_instructions}}'.\\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_PROMPT_INPUT \""
```

**Success:** False

---

## Meeting Notes Extraction

### Test 1
**Input:** Weekly team standup on 2024-01-15
Attendees: John, Sarah, Mike, Lisa
We discussed the new feature requirements, reviewed the sprint progress, and planned the next release. 
Action items: John to complete API documentation by Friday, Sarah to review the UI mockups, Mike to set up the testing environment.
Next meeting scheduled for January 22nd.

**Output:**
```json
"\"Input to ChatPromptTemplate is missing variables {'format_instructions'}.  Expected: ['format_instructions', 'input_text'] Received: ['input_text']\\nNote: if you intended {format_instructions} to be part of the string and not a variable, please escape it with double curly braces like: '{{format_instructions}}'.\\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_PROMPT_INPUT \""
```

**Success:** False

---

### Test 2
**Input:** Project kickoff meeting - 2024-01-10
Team members: Alice, Bob, Charlie, Diana
Main topics: Project scope, timeline, and resource allocation. We also discussed the technical architecture and risk assessment.
Tasks: Alice to create project timeline, Bob to research third-party integrations, Charlie to set up development environment, Diana to prepare stakeholder presentations.
No next meeting scheduled yet.

**Output:**
```json
"\"Input to ChatPromptTemplate is missing variables {'format_instructions'}.  Expected: ['format_instructions', 'input_text'] Received: ['input_text']\\nNote: if you intended {format_instructions} to be part of the string and not a variable, please escape it with double curly braces like: '{{format_instructions}}'.\\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_PROMPT_INPUT \""
```

**Success:** False

---

## Key Benefits

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

