import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama


def main() -> None:
    # Choose a local model you've pulled with `ollama pull <model>`
    # Common choices: "llama3", "mistral", "qwen2"
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:12b-it-qat")

    llm = Ollama(model=model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a friendly assistant who answers concisely."),
            ("human", "Say hello to Nova and introduce yourself in one short sentence."),
        ]
    )

    # LCEL: prompt | llm
    chain = prompt | llm

    result = chain.invoke({})
    print(result)


if __name__ == "__main__":
    main()


