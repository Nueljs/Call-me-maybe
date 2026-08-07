import json
import numpy as np
from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore


class Generator:
    def __init__(self, llm: Small_LLM_Model) -> None:
        self.llm: Small_LLM_Model = llm
        self.vocab: dict[str, int] = self._load_vocab()

    def _load_vocab(self) -> dict[str, int]:
        """Loads the model's vocabulary and returns it as a dictionary"""
        path_vocab: str = self.llm.get_path_to_vocab_file()
        with open(path_vocab, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        input_ids_tensor: Any = self.llm.encode(prompt)
        input_ids: list[int] = input_ids_tensor[0].tolist()
        logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
        next_token_id: int = int(np.argmax(logits))
        input_ids.append(next_token_id)

        return self.llm.decode(input_ids)
