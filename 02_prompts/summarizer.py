import json
import os
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.llms import Ollama


class SummarySchema(BaseModel):
    title: str = Field(description="Short, descriptive title")
    summary: str = Field(description="2-3 sentence summary")
    keywords: list[str] = Field(description="3-6 important keywords")


def build_prompt(parser: JsonOutputParser) -> ChatPromptTemplate:
    examples = [
        {
            "input": "LangChain helps developers build with LLMs by providing chains, tools, and memory.",
            "output": {
                "title": "LangChain in a Nutshell",
                "summary": "LangChain is a framework for composing LLM apps. It offers chains to sequence steps, memory for context, and tool integrations.",
                "keywords": ["LangChain", "LLM", "chains", "memory", "tools"],
            },
        },
        {
            "input": "Ollama runs LLMs locally so developers can experiment without cloud APIs.",
            "output": {
                "title": "Local LLMs with Ollama",
                "summary": "Ollama lets you run models on your machine for quick prototyping and privacy. Pull a model and prompt it locally.",
                "keywords": ["Ollama", "local models", "privacy", "prototyping"],
            },
        },
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Text: {input}"),
            ("ai", json.dumps({"title": "{title}", "summary": "{summary}", "keywords": ["{k1}", "{k2}"]})),
        ]
    )

    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=[
            {
                "input": e["input"],
                "title": e["output"]["title"],
                "summary": e["output"]["summary"],
                "k1": e["output"]["keywords"][0],
                "k2": e["output"]["keywords"][1],
            }
            for e in examples
        ],
    )

    system = (
        "You are a concise summarizer. Produce valid JSON only with keys: title, summary, keywords. "
        f"Follow this JSON schema: {parser.get_format_instructions()}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            few_shot,
            ("human", "Summarize this text:\n{input_text}"),
        ]
    )
    return prompt


def main() -> None:
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:12b-it-qat")
    llm = Ollama(model=model_name)

    parser = JsonOutputParser(pydantic_object=SummarySchema)
    prompt = build_prompt(parser)

    chain = prompt | llm | parser

    input_text = (
        "LangChain standardizes common LLM app patterns like prompts, chains, memory, and tools, "
        "so developers can build quickly and swap models easily."
    )

    result = chain.invoke({"input_text": input_text})

    out_path = Path(__file__).parent / "summary.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


