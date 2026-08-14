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
        self.fn_map: dict[str, FunctionDefinition] = {
            f.name: f for f in self.functions
        }

        for fn_names in self.functions_names:
            tokens = self.llm.encode(fn_names)[0].tolist()
            self.fn_token_paths.append(tokens)

    def _load_vocab(self) -> dict[str, int]:
        """Loads the model's vocabulary and returns it as a dictionary"""
        path_vocab: str = self.llm.get_path_to_vocab_file()
        with open(path_vocab, 'r', encoding='utf-8') as f:
            data: dict[str, int] = json.load(f)
            return data

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        input_ids_tensor: Any = self.llm.encode(prompt)
        original_prompt_id: list[int] = input_ids_tensor[0].tolist()
        prompt_length: int = len(original_prompt_id)
        input_ids: list[int] = input_ids_tensor[0].tolist()

        open_brackets: int = 0
        current_state: int = 0

        id_key: int = self.vocab["{"]
        ids_name_key: list[int] = self.llm.encode('\n "name": ')[0].tolist()
        name_pointer: int = 0

        ids_params_key: list[int] = self.llm.encode(
            '\n "parameters": {')[0].tolist()
        params_pointer: int = 0

        active_paths: list[list[int]] = [
            path.copy() for path in self.fn_token_paths
        ]

        selected_fn_name: str = ""
        params_keys_tokens: list[list[int]] = []
        current_param_idx: int = 0
        params_key_pointer: int = 0
        my_turn: bool = True
        inside_str: bool = False

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

            elif current_state == 3:
                mask4 = np.full_like(logits, -np.inf)
                mask4[ids_params_key[params_pointer]] = 0.0
                logits_array = mask4

            elif current_state == 4:
                if my_turn and current_param_idx < len(params_keys_tokens):
                    mask5 = np.full_like(logits, -np.inf)
                    mask5[params_keys_tokens[
                        current_param_idx][
                        params_key_pointer]] = 0.0
                    logits_array = mask5
                else:
                    mask5 = np.array(logits)
                    mask5[id_key] = -np.inf
                    logits_array = mask5

            next_token_id: int = int(np.argmax(logits_array))
            token_text: str = self.llm.decode([next_token_id])

            open_brackets = open_brackets + token_text.count('{')
            open_brackets = open_brackets - token_text.count('}')
            input_ids.append(next_token_id)

            was_inside = inside_str

            if '"' in token_text:
                if token_text.count('"') % 2 != 0:
                    inside_str = not inside_str

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
                    previous_text = self.llm.decode(input_ids)

                    for fn_name in self.fn_map:
                        if f'"{fn_name}"' in previous_text:
                            selected_fn_name = fn_name
                            break

                    current_state = 3

            elif current_state == 3:
                params_pointer = params_pointer + 1

                if params_pointer == len(ids_params_key):
                    current_state = 4

                    if selected_fn_name in self.fn_map:
                        curr_fun = self.fn_map[selected_fn_name]

                        properties = curr_fun.parameters.get("properties", {})

                        for i, param_name in enumerate(properties.keys()):
                            prefix = ' "' if i == 0 else ', "'
                            forced_text = f'{prefix}{param_name}": '

                            forced_tokens = self.llm.encode(
                                forced_text)[0].tolist()
                            params_keys_tokens.append(forced_tokens)

            elif current_state == 4:
                if my_turn == 4:
                    params_key_pointer = params_key_pointer + 1
                    if params_key_pointer == len(params_keys_tokens[
                       current_param_idx]):
                        my_turn = False
                        params_key_pointer = 0
                        current_param_idx = current_param_idx + 1
                else:
                    just_closed_string = (was_inside and not inside_str)
                    comma_outside = (not inside_str and ',' in token_text)

                    if just_closed_string:
                        my_turn = True
                    elif comma_outside:
                        input_ids.pop()
                        my_turn = True

            if current_state > 0 and open_brackets == 0:
                break

        generated_ids: list[int] = input_ids[prompt_length:]
        final_result: str = str(self.llm.decode(generated_ids))

        return final_result
