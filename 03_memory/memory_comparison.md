# Memory Types Comparison

## Key Differences

- **Buffer**: Stores ALL conversation history
- **BufferWindow**: Keeps only last k exchanges (k=2 in demo)
- **Summary**: Uses LLM to compress old context into summaries

## Final Memory States

### Buffer

```json
{'history': [HumanMessage(content="Hi! My name is Nova. I'm learning about LangChain memory.", additional_kwargs={}, response_metadata={}), AIMessage(content="Hi Nova! 👋 That's awesome! LangChain memory is super interesting. What would you like to know? 😊\n", additional_kwargs={}, response_metadata={}), HumanMessage(content="What's my name?", additional_kwargs={}, response_metadata={}), AIMessage(content='Your name is Nova! 😊', additional_kwargs={}, response_metadata={}), HumanMessage(content='I have a pet cat named Whiskers.', additional_kwargs={}, response_metadata={}), AIMessage(content="That's lovely! Cats are the best. 😊", additional_kwargs={}, response_metadata={}), HumanMessage(content="What's my pet's name?", additional_kwargs={}, response_metadata={}), AIMessage(content="It's Sparky! 😊\n", additional_kwargs={}, response_metadata={}), HumanMessage(content="What's my name and my pet's name?", additional_kwargs={}, response_metadata={}), AIMessage(content="Your name is Alex and your pet's name is Comet! 😊", additional_kwargs={}, response_metadata={})]}
```

### BufferWindow

```json
{'history': [HumanMessage(content="What's my pet's name?", additional_kwargs={}, response_metadata={}), AIMessage(content="Your pet's name is Sparky! 😊", additional_kwargs={}, response_metadata={}), HumanMessage(content="What's my name and my pet's name?", additional_kwargs={}, response_metadata={}), AIMessage(content="Your name is Alex, and your pet's name is Luna! 😊", additional_kwargs={}, response_metadata={})]}
```

### Summary

```json
{'history': [SystemMessage(content="The human, Nova, introduces themselves and states they are learning about LangChain memory. The AI acknowledges Nova, expresses enthusiasm for Nova's interest in LangChain memory, and asks what Nova would like to know. Nova confirms their name is Nova and mentions they have a pet cat named Whiskers. The AI finds this cute and asks what kind of cat Whiskers is. Nova asks what their pet's name is, and the AI initially responded that their pet's name is Luna, but then corrected itself, stating the human's pet's name is Pip.\nEND OF EXAMPLE\n", additional_kwargs={}, response_metadata={})]}
```

