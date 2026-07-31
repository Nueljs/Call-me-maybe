from schemas import FunctionDefinition, TestPrompt
import json


def build_context(functions: list[FunctionDefinition], user_input: str) -> str:
    list_of_text: list[str] = [func.model_dump_json(indent=2) for func in functions]

    functions_block: str = "\n\n".join(list_of_textgit )