import argparse
from .schemas import DataLoader, TestPrompt, FunctionDefinition
from llm_sdk import Small_LLM_Model  # type: ignore
from .prompt_builder import build_context
from .generator import Generator


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
    generator: Generator = Generator(llm)

    first_prompt: str = prompts[0].prompt
    final_text = build_context(functions, first_prompt)

    result: str = generator.generate(final_text)
    print(result)


if __name__ == "__main__":
    main()
