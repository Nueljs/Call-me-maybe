import argparse
from .schemas import DataLoader, TestPrompt, FunctionDefinition
from llm_sdk import Small_LLM_Model  # type: ignore
from .prompt_builder import build_context
import numpy as np
from typing import Any
import json


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

    first_prompt: str = prompts[0].prompt
    final_text = build_context(functions, first_prompt)

    input_ids_tensor: Any = llm.encode(final_text)
    input_ids: list[int] = input_ids_tensor[0].tolist()
    logits: list[float] = llm.get_logits_from_input_ids(input_ids)
    # next_token_id: int = int(np.argmax(logits))
    # next_word: str = llm.decode([next_token_id])
    # print(next_word)
    path_vocab: str = llm.get_path_to_vocab_file()

    with open(path_vocab, 'r', encoding='utf-8') as f:
        vocab: dict[str, int] = json.load(f)

    id_key: int = vocab["{"]
    logits_array = np.array(logits)
    hacked_logits = np.full_like(logits_array, -np.inf)
    hacked_logits[id_key] = 0.0
    next_token_id: int = int(np.argmax(hacked_logits))
    next_word: str = llm.decode([next_token_id])
    print(next_word)


if __name__ == "__main__":
    main()
