import json
import numpy as np
from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore
from .schemas import FunctionDefinition
from enum import Enum


class States(Enum):
    START = 1
    NAME_KEY = 2
    FUNC_NAME = 3
    PARAM_KEY = 4
    PARAMS = 5
    END = 6


class Generator:
    """Class responsible for generating constrained JSON function calls."""
    def __init__(self,
                 llm: Small_LLM_Model,
                 functions: list[FunctionDefinition]) -> None:
        """Initialize the Generator with the LLM and target functions."""
        self.llm: Small_LLM_Model = llm
        self.vocab: dict[str, int] = self._load_vocab()
        self.functions: list[FunctionDefinition] = functions
        self.functions_names: list[str] = [
            f'"{function.name}",' for function in self.functions
        ]
        self.fn_token_paths: list[tuple[int, list[int]]] = []

        for index, fn_names in enumerate(self.functions_names):
            tokens = self.llm.encode(fn_names)[0].tolist()
            self.fn_token_paths.append((index, tokens))

        self.name_key: list[int] = self.llm.encode('"name": ')[0].tolist()
        self.param_key: list[int] = self.llm.encode(
            '"parameters": {')[0].tolist()

    def _load_vocab(self) -> dict[str, int]:
        """Loads the model's vocabulary and returns it as a dictionary"""
        path_vocab: str = self.llm.get_path_to_vocab_file()
        with open(path_vocab, 'r', encoding='utf-8') as f:
            data: dict[str, int] = json.load(f)
            return data

    def _advance_func_paths(self,
                            active_paths: list[tuple[int, list[int]]],
                            token: int) -> list[tuple[int, list[int]]]:
        new_active_paths: list[tuple[int, list[int]]] = []

        for path in active_paths:
            if path[1] and path[1][0] == token:
                new_active_paths.append((path[0], path[1][1:]))

        return new_active_paths

    def _get_allowed_tokens(self, state: States,
                            active_paths: list[
                                tuple[int, list[int]]] | None = None
                            ) -> set[int]:
        allowed_tokens: set[int] = set()
        if state == States.START:
            allowed_tokens.add(self.vocab["{"])

        elif state == States.FUNC_NAME:
            if active_paths is None:
                raise ValueError("active_paths required in FUNC_NAME")
            for path in active_paths:
                if path[1]:
                    allowed_tokens.add(path[1][0])

        return allowed_tokens

    def _has_finished_path(self,
                           active_paths: list[tuple[int, list[int]]]) -> bool:
        return any(not path[1] for path in active_paths)

    def _get_next_token(self, input_ids: list[int],
                        allowed_tokens: set[int]) -> int:
        logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
        logits_array = np.array(logits)
        mask = np.full_like(logits_array, -np.inf)

        for token_id in allowed_tokens:
            mask[token_id] = logits_array[token_id]

        return int(np.argmax(mask))

    def generate(self,
                 prompt: str,
                 max_tokens: int = 200) -> str:

        state: States = States.START
        input_ids: list[int] = self.llm.encode(prompt)[0].tolist()

        allowed_tokens: set[int] = self._get_allowed_tokens(
            state)
        next_token: int = self._get_next_token(input_ids, allowed_tokens)

        input_ids.append(next_token)

        current_fixed_path: list[int] = []

        state = States.NAME_KEY
        current_fixed_path = self.name_key.copy()
        while state == States.NAME_KEY:
            input_ids.append(current_fixed_path[0])
            current_fixed_path = current_fixed_path[1:]

            if not current_fixed_path:
                state = States.FUNC_NAME

        active_paths: list[tuple[int, list[int]]] = [
            (index, path.copy())
            for index, path in self.fn_token_paths]

        while state == States.FUNC_NAME:
            allowed_tokens = self._get_allowed_tokens(state, active_paths)
            next_token = self._get_next_token(input_ids, allowed_tokens)
            input_ids.append(next_token)
            active_paths = self._advance_func_paths(active_paths, next_token)
            print(active_paths)

            if self._has_finished_path(active_paths):
                state = States.PARAM_KEY

        current_fixed_path = self.param_key.copy()
        while state == States.PARAM_KEY:
            input_ids.append(current_fixed_path[0])
            current_fixed_path = current_fixed_path[1:]

            if not current_fixed_path:
                state = States.PARAMS

        return self.llm.decode(input_ids)

    # def old_generate(self, prompt: str, max_tokens: int = 200) -> str:
    #     """Generate a constrained function call based on the prompt.

    #     Args:
    #         prompt (str): The natural language input from the user.
    #         max_tokens (int): The maximum number of tokens to generate.
    #     Returns:
    #         str: A raw string containing the generated JSON.
    #     """
    #     input_ids_tensor: Any = self.llm.encode(prompt)
    #     original_prompt_id: list[int] = input_ids_tensor[0].tolist()
    #     prompt_length: int = len(original_prompt_id)
    #     input_ids: list[int] = input_ids_tensor[0].tolist()

    #     open_brackets: int = 0
    #     current_state: int = 0

    #     id_key: int = self.vocab["{"]
    #     ids_name_key: list[int] = self.llm.encode('\n "name": ')[0].tolist()
    #     name_pointer: int = 0

    #     ids_params_key: list[int] = self.llm.encode(
    #         '\n "parameters": {')[0].tolist()
    #     params_pointer: int = 0

    #     active_paths: list[list[int]] = [
    #         path.copy() for path in self.fn_token_paths
    #     ]

    #     for _ in range(max_tokens):
    #         logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
    #         logits_array = np.array(logits)
    #         if current_state == 0:
    #             mask = np.full_like(logits, -np.inf)
    #             mask[id_key] = 0.0
    #             logits_array = mask

    #         elif current_state == 1:
    #             mask2 = np.full_like(logits, -np.inf)
    #             mask2[ids_name_key[name_pointer]] = 0
    #             logits_array = mask2

    #         elif current_state == 2:
    #             mask3 = np.full_like(logits, -np.inf)
    #             for path in active_paths:
    #                 mask3[path[0]] = logits_array[path[0]]

    #             logits_array = mask3

    #         elif current_state == 3:
    #             mask4 = np.full_like(logits, -np.inf)
    #             mask4[ids_params_key[params_pointer]] = 0.0
    #             logits_array = mask4

    #         next_token_id: int = int(np.argmax(logits_array))
    #         token_text: str = self.llm.decode([next_token_id])

    #         open_brackets = open_brackets + token_text.count('{')
    #         open_brackets = open_brackets - token_text.count('}')
    #         input_ids.append(next_token_id)

    #         if current_state == 0:
    #             if "{" in token_text:
    #                 current_state = 1

    #         elif current_state == 1:
    #             name_pointer = name_pointer + 1
    #             if name_pointer == len(ids_name_key):
    #                 current_state = 2

    #         elif current_state == 2:
    #             new_paths: list[list[int]] = []

    #             for path in active_paths:
    #                 if path[0] == next_token_id:
    #                     new_paths.append(path[1:])

    #             active_paths = new_paths

    #             if len(active_paths[0]) == 0:
    #                 current_state = 3

    #         elif current_state == 3:
    #             params_pointer = params_pointer + 1

    #             if params_pointer == len(ids_params_key):
    #                 current_state = 4

    #         if current_state > 0 and open_brackets == 0:
    #             break

    #     generated_ids: list[int] = input_ids[prompt_length:]

    #     return str(self.llm.decode(generated_ids))
