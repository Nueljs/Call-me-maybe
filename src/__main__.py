import argparse
import json
import os

from typing import Any

from .schemas import (
    DataLoader,
    TestPrompt,
    FunctionDefinition,
    OutputResult
)
from llm_sdk import Small_LLM_Model  # type: ignore
from .prompt_builder import build_context
from .generator import Generator


def main() -> None:
    """Execute the main flow of the prompt processing pipeline."""
    parser = argparse.ArgumentParser(description='Processing args')
    parser.add_argument(
        '--input',
        type=str,
        default="data/input/function_calling_tests.json",
        help='The input file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default="data/output/function_calling_results.json",
        help='The output file'
    )
    parser.add_argument(
        '--functions_definition',
        type=str,
        default="data/input/functions_definition.json",
        help='The functions file'
    )

    args = parser.parse_args()

    input_path: str = str(args.input)
    output_path: str = str(args.output)
    function_path: str = str(args.functions_definition)

    prompt_loader: DataLoader = DataLoader(input_path)
    prompts: list[TestPrompt] = prompt_loader.prompt_request()

    function_loader: DataLoader = DataLoader(function_path)
    functions: list[FunctionDefinition] = function_loader.function_request()

    llm: Small_LLM_Model = Small_LLM_Model()
    generator: Generator = Generator(llm, functions)

    final_result: list[dict[str, Any]] = []

    for prompt_item in prompts:
        final_text = build_context(functions, prompt_item.prompt)
        result_str: str = generator.generate(final_text)

        try:
            generated_dict: dict[str, Any] = json.loads(result_str)

            final_obj: OutputResult = OutputResult(
                prompt=prompt_item.prompt,
                name=generated_dict["name"],
                parameters=generated_dict["parameters"]
            )

            final_result.append(final_obj.model_dump())

            print(
                f"Processed: '{prompt_item.prompt}' -> "
                f"{generated_dict['name']}"
            )

        except json.JSONDecodeError:
            print(
                f"Error processing the prompt: '{prompt_item.prompt}'"
            )
            print(f"Raw-text: {result_str}")

        except Exception as e:
            print(
                f"Validation error in the prompt: "
                f"'{prompt_item.prompt}' - {e}"
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(final_result, f, indent=2)


if __name__ == "__main__":
    main()
