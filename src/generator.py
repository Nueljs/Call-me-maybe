import json
import numpy as np
from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore
from .schemas import FunctionDefinition


class Generator:
    def __init__(self,
                 llm: Small_LLM_Model,
                 functions: list[FunctionDefinition]) -> None:
        self.llm: Small_LLM_Model = llm
        self.vocab: dict[str, int] = self._load_vocab()
        self.functions: list[FunctionDefinition] = functions
        self.functions_names: list[str] = [
            f'"{function.name}",' for function in self.functions
        ]
        self.fn_token_paths: list[list[int]] = []

        for fn_names in self.functions_names:
            tokens = self.llm.encode(fn_names)[0].tolist()
            self.fn_token_paths.append(tokens)

    def _load_vocab(self) -> dict[str, int]:
        """Loads the model's vocabulary and returns it as a dictionary"""
        path_vocab: str = self.llm.get_path_to_vocab_file()
        with open(path_vocab, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        input_ids_tensor: Any = self.llm.encode(prompt)
        input_ids: list[int] = input_ids_tensor[0].tolist()

        open_brackets: int = 0
        current_state: int = 0

        id_key: int = self.vocab["{"]
        ids_name_key = self.llm.encode('\n "name": ')[0].tolist()
        name_pointer: int = 0

        active_paths: list[list[int]] = [
            path.copy() for path in self.fn_token_paths
        ]

        for _ in range(max_tokens):
            logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
            logits_array = np.array(logits)
            if current_state == 0:
                mask = np.full_like(logits, -np.inf)
                mask[id_key] = 0.0
                logits_array = mask

            elif current_state == 1:
                mask2 = np.full_like(logits, -np.inf)
                mask2[ids_name_key[name_pointer]] = 0
                logits_array = mask2

            elif current_state == 2:
                mask3 = np.full_like(logits, -np.inf)
                for path in active_paths:
                    mask3[path[0]] = logits_array[path[0]]

                logits_array = mask3

            next_token_id: int = int(np.argmax(logits_array))
            token_text: str = self.llm.decode([next_token_id])

            open_brackets = open_brackets + token_text.count('{')
            open_brackets = open_brackets - token_text.count('}')
            input_ids.append(next_token_id)

            if current_state == 0:
                if "{" in token_text:
                    current_state = 1

            elif current_state == 1:
                name_pointer = name_pointer + 1
                if name_pointer == len(ids_name_key):
                    current_state = 2

            elif current_state == 2:
                new_paths: list[list[int]] = []

                for path in active_paths:
                    if path[0] == next_token_id:
                        new_paths.append(path[1:])

                active_paths = new_paths

                if len(active_paths[0]) == 0:
                    current_state = 3

            if current_state > 0 and open_brackets == 0:
                break

        return self.llm.decode(input_ids)
