from pydantic import BaseModel
import json
from typing import Any


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class TestPrompt(BaseModel):
    prompt: str


class DataLoader:
    def __init__(self, path: str) -> None:
        self.path: str = path

    def prompt_request(self) -> list[TestPrompt]:
        with open(self.path, 'r', encoding='utf-8') as file:
            raw_prompts: list[dict[str, str]] = json.load(file)
            prompts: list[TestPrompt] = []

            for item in raw_prompts:
                prompts.append(TestPrompt(**item))

        return prompts

    def function_request(self) -> list[FunctionDefinition]:
        with open(self.path, 'r', encoding='utf-8') as file:
            raw_functions: dict[str, Any] = json.load(file)
            functions: list[FunctionDefinition] = []

            for item in raw_functions:
                functions.append(FunctionDefinition(**raw_functions))

        return functions
