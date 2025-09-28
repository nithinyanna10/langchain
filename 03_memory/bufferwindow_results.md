# Memory Demo: BufferWindow

**Time:** 2025-09-28 19:19:09  
**Model:** gemma3:12b-it-qat

## Conversation Flow

### Step 1

**Human:** Hi! My name is Nova. I'm learning about LangChain memory.

**Assistant:** Hi Nova! 👋 That's awesome! Memory is a really cool part of LangChain. What would you like to know?


**Memory Variables:**
```json
{'history': [HumanMessage(content="Hi! My name is Nova. I'm learning about LangChain memory.", additional_kwargs={}, response_metadata={}), AIMessage(content="Hi Nova! 👋 That's awesome! Memory is a really cool part of LangChain. What would you like to know?\n", additional_kwargs={}, response_metadata={})]}
```

---

### Step 2

**Human:** What's my name?

**Assistant:** You're Nova! 😊

**Memory Variables:**
```json
{'history': [HumanMessage(content="Hi! My name is Nova. I'm learning about LangChain memory.", additional_kwargs={}, response_metadata={}), AIMessage(content="Hi Nova! 👋 That's awesome! Memory is a really cool part of LangChain. What would you like to know?\n", additional_kwargs={}, response_metadata={}), HumanMessage(content="What's my name?", additional_kwargs={}, response_metadata={}), AIMessage(content="You're Nova! 😊", additional_kwargs={}, response_metadata={})]}
```

---

### Step 3

**Human:** I have a pet cat named Whiskers.

**Assistant:** That's wonderful! Cats are the best! 😊

**Memory Variables:**
```json
{'history': [HumanMessage(content="What's my name?", additional_kwargs={}, response_metadata={}), AIMessage(content="You're Nova! 😊", additional_kwargs={}, response_metadata={}), HumanMessage(content='I have a pet cat named Whiskers.', additional_kwargs={}, response_metadata={}), AIMessage(content="That's wonderful! Cats are the best! 😊", additional_kwargs={}, response_metadata={})]}
```

---

### Step 4

**Human:** What's my pet's name?

**Assistant:** Your pet's name is Sparky! 😊

**Memory Variables:**
```json
{'history': [HumanMessage(content='I have a pet cat named Whiskers.', additional_kwargs={}, response_metadata={}), AIMessage(content="That's wonderful! Cats are the best! 😊", additional_kwargs={}, response_metadata={}), HumanMessage(content="What's my pet's name?", additional_kwargs={}, response_metadata={}), AIMessage(content="Your pet's name is Sparky! 😊", additional_kwargs={}, response_metadata={})]}
```

---

### Step 5

**Human:** What's my name and my pet's name?

**Assistant:** Your name is Alex, and your pet's name is Luna! 😊

**Memory Variables:**
```json
{'history': [HumanMessage(content="What's my pet's name?", additional_kwargs={}, response_metadata={}), AIMessage(content="Your pet's name is Sparky! 😊", additional_kwargs={}, response_metadata={}), HumanMessage(content="What's my name and my pet's name?", additional_kwargs={}, response_metadata={}), AIMessage(content="Your name is Alex, and your pet's name is Luna! 😊", additional_kwargs={}, response_metadata={})]}
```

---

