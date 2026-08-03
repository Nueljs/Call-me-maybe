import argparse
from .schemas import DataLoader, TestPrompt, FunctionDefinition
from llm_sdk import Small_LLM_Model  # type: ignore
from .prompt_builder import build_context
import numpy as np
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description='Processing args')
    parser.add_argument(
        '--input', type=str, default="data/input/function_calling_tests.json",
        help='The input file')
    parser.add_argument(
        '--output', type=str, default="data/output/function_calls.json",
        help='The output file')
    parser.add_argument(
        '--functions_definition', type=str,
        default="data/input/functions_definition.json",
        help='The functions file')
    args = parser.parse_args()

    prompt_loader: DataLoader = DataLoader(args.input)
    prompts: list[TestPrompt] = prompt_loader.prompt_request()
    function_loader: DataLoader = DataLoader(args.functions_definition)
    functions: list[FunctionDefinition] = function_loader.function_request()

    llm: Small_LLM_Model = Small_LLM_Model()

    # print(f"Exito hemos cargado {len(prompts)} prompts")
    # print(f"Exito hemos cargado {len(functions)} functions")
    # print("Motor LLM instaciado y listo para la accion")

    first_prompt: str = prompts[0].prompt
    final_text = build_context(functions, first_prompt)
    # print(final_text)
    input_ids_tensor: Any = llm.encode(final_text)
    print(input_ids_tensor)
    input_ids: list[int] = input_ids_tensor[0].tolist()
    logits: list[float] = llm.get_logits_from_input_ids(input_ids)
    next_token_id: int = int(np.argmax(logits))
    next_word: str = llm.decode([next_token_id])
    print(next_word)


if __name__ == "__main__":
    main()
