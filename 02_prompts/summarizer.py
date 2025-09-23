import json
import os
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:  # fallback if langchain-ollama not installed
    from langchain_community.llms import Ollama  # type: ignore


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

    # Avoid curly-brace collisions by passing pre-rendered JSON in a single variable
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Text: {input}"),
            ("ai", "{example_json}"),
        ]
    )

    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=[
            {
                "input": e["input"],
                "example_json": json.dumps(e["output"], ensure_ascii=False),
            }
            for e in examples
        ],
    )

    # IMPORTANT: Don't inline the JSON schema here, or `{}` inside it
    # will be mistaken for template variables. Inject via a single variable.
    system = (
        "You are a concise summarizer. Produce valid JSON only with keys: title, summary, keywords.\n"
        "Follow this JSON schema:\n{format_instructions}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            few_shot,
            ("human", "Summarize this text:\n{input_text}"),
        ]
    )
    # Inject the format instructions as a single variable to avoid brace-collisions
    return prompt.partial(format_instructions=parser.get_format_instructions())


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


