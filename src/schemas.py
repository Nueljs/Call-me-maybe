from pydantic import BaseModel
import json
from typing import Any
import os


class FunctionDefinition(BaseModel):
    """Schema representing a function definition available to the LLM."""
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class TestPrompt(BaseModel):
    """Schema for test input prompts."""
    prompt: str


class OutputResult(BaseModel):
    """Schema validating the structured output before saving."""
    prompt: str
    name: str
    parameters: dict[str, Any]


class DataLoader:
    """
    Handles loading and parsing of input JSON files with
    strict error handling
    """
    def __init__(self, path: str) -> None:
        """Initialize DataLoader with path to a JSON file"""
        self.path: str = path

    def prompt_request(self) -> list[TestPrompt]:
        """Loads test prompts from a JSON file safely
            Returns:
            list[TestPrompts]: Parsed prompts, or an empty list
            if loading fails.
        """
        if not os.path.exists(self.path):
            print(f"Error: Prompt file not found at {self.path}")
            return []

        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                raw_prompts: list[dict[str, Any]] = json.load(file)
                prompts: list[TestPrompt] = []

                for item in raw_prompts:
                    prompts.append(TestPrompt(**item))

                return prompts

        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading prompts from '{self.path}': {e}")
            return []

    def function_request(self) -> list[FunctionDefinition]:
        """Loads function definitions from a JSON file safely.

        Returns:
            list[FunctionDefinition]: Parsed function definitions,
            or empty list on failure.
        """
        if not os.path.exists(self.path):
            print(f"Error: Function definitions file not"
                  f" found at '{self.path}'")
            return []
        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                raw_functions: list[dict[str, Any]] = json.load(file)
                functions: list[FunctionDefinition] = []

                for item in raw_functions:
                    functions.append(FunctionDefinition(**item))
            return functions
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading functions from '{self.path}': {e}")
            return []
