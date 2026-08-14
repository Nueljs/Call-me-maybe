import json
import numpy as np
from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore
from .schemas import FunctionDefinition


class Generator:
    """Class responsible for generating constrained function calls."""

    def __init__(self,
                 llm: Small_LLM_Model,
                 functions: list[FunctionDefinition]) -> None:
        """Initialize the Generator with a model and function definitions."""
        self.llm: Small_LLM_Model = llm
        self.functions: list[FunctionDefinition] = functions
        self.vocab: dict[str, int] = self._load_vocab()

        self.fn_map: dict[str, FunctionDefinition] = {
            f.name: f for f in self.functions
        }

        self.fn_token_paths: list[list[int]] = []
        for function in self.functions:
            tokens: list[int] = self.llm.encode(function.name)[0].tolist()
            self.fn_token_paths.append(tokens)

        self.number_tokens: set[int] = self._get_number_tokens()

    def _load_vocab(self) -> dict[str, int]:
        """Load the model's vocabulary and return it as a dictionary."""
        path_vocab: str = self.llm.get_path_to_vocab_file()
        with open(path_vocab, 'r', encoding='utf-8') as f:
            data: dict[str, int] = json.load(f)
            return data

    def _get_number_tokens(self) -> set[int]:
        """Pre-calculate all valid token IDs for number generation.

        Returns:
            set[int]: A set of token IDs that contain only numbers, spaces,
            or valid decimal/negative signs.
        """
        valid_ids: set[int] = set()
        for text, t_id in self.vocab.items():
            clean_text: str = text.replace('Ġ', '').replace(' ', '').strip()

            if not clean_text:
                valid_ids.add(t_id)
                continue

            if all(c in "0123456789.-" for c in clean_text):
                valid_ids.add(t_id)
        return valid_ids

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate a constrained JSON function call.

        Args:
            prompt (str): The input natural language prompt.
            max_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: A valid JSON string representing the function call.
        """
        input_ids_tensor: Any = self.llm.encode(prompt)
        input_ids: list[int] = input_ids_tensor[0].tolist()
        prompt_length: int = len(input_ids)

        STATE_START: int = 0
        STATE_NAME: int = 1
        STATE_PARAMS_START: int = 2
        STATE_PARAM_KEY: int = 3
        STATE_PARAM_VALUE: int = 4
        STATE_PARAM_TERMINAL: int = 5
        STATE_END: int = 6

        current_state: int = STATE_START

        target_tokens: list[int] = self.llm.encode('{\n  "name": "')[0].tolist()
        target_pointer: int = 0

        active_paths: list[list[int]] = [
            path.copy() for path in self.fn_token_paths
        ]
        selected_fn: FunctionDefinition | None = None

        params_keys: list[str] = []
        current_param_idx: int = 0

        for _ in range(max_tokens):
            logits: list[float] = self.llm.get_logits_from_input_ids(input_ids)
            logits_array = np.array(logits, dtype=np.float32)
            mask = np.full_like(logits_array, -np.inf)

            if current_state in (STATE_START, STATE_PARAMS_START,
                                 STATE_PARAM_KEY, STATE_PARAM_TERMINAL,
                                 STATE_END):
                if target_pointer < len(target_tokens):
                    expected_token: int = target_tokens[target_pointer]
                    mask[expected_token] = 0.0

            elif current_state == STATE_NAME:
                for path in active_paths:
                    if len(path) > 0:
                        mask[path[0]] = logits_array[path[0]]

            elif current_state == STATE_PARAM_VALUE:
                if selected_fn is None:
                    break

                param_name: str = params_keys[current_param_idx]
                param_type: str = selected_fn.parameters[param_name].get(
                    "type", "string"
                )

                is_last: bool = current_param_idx == len(params_keys) - 1
                terminal_str: str = "}" if is_last else ","

                if param_type == "number":
                    for t_id in self.number_tokens:
                        mask[t_id] = logits_array[t_id]

                    term_id = self.vocab.get(terminal_str)
                    if term_id is not None:
                        mask[term_id] = logits_array[term_id]

                elif param_type == "string":
                    mask = logits_array.copy()

            next_token_id: int = int(np.argmax(mask))
            token_text: str = self.llm.decode([next_token_id])
            input_ids.append(next_token_id)

            if current_state == STATE_START:
                target_pointer += 1
                if target_pointer >= len(target_tokens):
                    current_state = STATE_NAME
                    active_paths = [
                        path.copy() for path in self.fn_token_paths]

            elif current_state == STATE_NAME:
                active_paths = [
                    p[1:] for p in active_paths if p[0] == next_token_id
                ]
                if len(active_paths) == 1 and len(active_paths[0]) == 0:
                    gen_str: str = self.llm.decode(input_ids[prompt_length:])

                    for fn_name in self.fn_map.keys():
                        if fn_name in gen_str:
                            selected_fn = self.fn_map[fn_name]
                            break

                    target_tokens = self.llm.encode(
                        '",\n  "parameters": {'
                    )[0].tolist()
                    target_pointer = 0
                    current_state = STATE_PARAMS_START
                    if selected_fn:
                        params_keys = list(selected_fn.parameters.keys())

            elif current_state == STATE_PARAMS_START:
                target_pointer += 1
                if target_pointer >= len(target_tokens):
                    if not params_keys:
                        target_tokens = self.llm.encode('\n}')[0].tolist()
                        target_pointer = 0
                        current_state = STATE_END
                    else:
                        current_param_idx = 0
                        if selected_fn:
                            p_type = selected_fn.parameters[
                                params_keys[0]].get(
                                "type", "string"
                            )
                            key_str = f'\n    "{params_keys[0]}": '
                            if p_type == "string":
                                key_str += '"'
                            target_tokens = self.llm.encode(key_str)[0].tolist()
                        target_pointer = 0
                        current_state = STATE_PARAM_KEY

            elif current_state == STATE_PARAM_KEY:
                target_pointer += 1
                if target_pointer >= len(target_tokens):
                    current_state = STATE_PARAM_VALUE

            elif current_state == STATE_PARAM_VALUE:
                is_last = current_param_idx == len(params_keys) - 1
                terminal_str = "}" if is_last else ","

                if selected_fn:
                    param_type = selected_fn.parameters[
                        params_keys[current_param_idx]
                    ].get("type", "string")

                    if param_type == "number":
                        if terminal_str in token_text:
                            if is_last:
                                break
                            else:
                                current_param_idx += 1
                                p_type = selected_fn.parameters[
                                    params_keys[current_param_idx]
                                ].get("type", "string")
                                key_str = f'\n    "{
                                    params_keys[current_param_idx]}": '
                                if p_type == "string":
                                    key_str += '"'
                                target_tokens = self.llm.encode(
                                    key_str)[0].tolist()
                                target_pointer = 0
                                current_state = STATE_PARAM_KEY

                    elif param_type == "string":
                        if '"' in token_text:
                            target_tokens = self.llm.encode(
                                terminal_str)[0].tolist()
                            target_pointer = 0
                            current_state = STATE_PARAM_TERMINAL

            elif current_state == STATE_PARAM_TERMINAL:
                target_pointer += 1
                if target_pointer >= len(target_tokens):
                    if current_param_idx == len(params_keys) - 1:
                        break
                    else:
                        current_param_idx += 1
                        if selected_fn:
                            p_type = selected_fn.parameters[
                                params_keys[current_param_idx]
                            ].get("type", "string")
                            key_str = f'\n    "{
                                params_keys[current_param_idx]}": '
                            if p_type == "string":
                                key_str += '"'
                            target_tokens = self.llm.encode(key_str)[0].tolist()
                            target_pointer = 0
                            current_state = STATE_PARAM_KEY

            elif current_state == STATE_END:
                target_pointer += 1
                if target_pointer >= len(target_tokens):
                    break

        generated_ids: list[int] = input_ids[prompt_length:]
        return str(self.llm.decode(generated_ids))