from pydantic import BaseModel
import json


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class TestPrompt(BaseModel):
    prompt: str


class DataLoader:
    def __init__