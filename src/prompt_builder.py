from .schemas import FunctionDefinition


def build_context(functions: list[FunctionDefinition], user_input: str) -> str:
    list_of_text: list[str] = [
        func.model_dump_json(indent=2) for func in functions]

    functions_block: str = "\n\n".join(list_of_text)

    final_prompt: str = f"""You are an expert analyst. You have access to the
    following tools:{functions_block}
    The user said: "{user_input}"
    Generate a call to the correct function in json format:"""

    return final_prompt
