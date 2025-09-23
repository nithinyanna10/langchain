import os
from datetime import datetime
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama


def build_chain(model_name: str):
    llm = Ollama(model=model_name)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a friendly teacher. Answer clearly in 2-4 sentences."),
            ("human", "Question: {question}"),
        ]
    )
    return prompt | llm


def save_markdown(output_path: Path, question: str, answer: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = (
        f"# Q&A Result\n\n"
        f"- Time: {timestamp}\n"
        f"- Model: {os.environ.get('OLLAMA_MODEL', 'gemma3:12b-it-qat')}\n\n"
        f"## Question\n{question}\n\n"
        f"## Answer\n{answer}\n"
    )
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:12b-it-qat")
    chain = build_chain(model_name)

    question = (
        "Explain LangChain chains like I'm 5, in a tiny story with one example."
    )

    answer = chain.invoke({"question": question})

    output_file = Path(__file__).parent / "qa_result.md"
    save_markdown(output_file, question, str(answer))
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()


