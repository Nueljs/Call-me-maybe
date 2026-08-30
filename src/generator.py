import json
import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore
from .schemas import FunctionDefinition
from enum import Enum


class States(Enum):
    START = 1
    NAME_KEY = 2
    FUNC_NAME = 3
    PARAM_KEY = 4
    PARAM_NAME = 5
    PARAM_VALUE = 6
    END = 7


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

        self.name_key: list[int] = self.llm.encode('\n "name": ')[0].tolist()
        self.param_key: list[int] = self.llm.encode(
            '\n "parameters": {')[0].tolist()

    def _load_vocab(self) -> dict[str, int]:
        """Loads the model's vocabulary and returns it as a dictionary"""
        path_vocab: str = self.llm.get_path_to_vocab_file()
        with open(path_vocab, 'r', encoding='utf-8') as f:
            data: dict[str, int] = json.load(f)
            return data

    def _selected_func(self, active_paths: list[tuple[int, list[int]]]) -> int:
        for func in active_paths:
            if len(func[1]) == 0:
                return func[0]

        raise ValueError("No function path has finished")

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

    def _get_allowed_param_tokens(self,
                                  param_paths: list[
                                      tuple[str, list[int]]]) -> set[int]:
        allowed_tokens: set[int] = set()

        for param in param_paths:
            if param[1]:
                allowed_tokens.add(param[1][0])

        return allowed_tokens

    def _has_finished_path(self,
                           active_paths: list[tuple[int, list[int]]]) -> bool:
        return any(not path[1] for path in active_paths)

    def _get_number_end_tokens(self) -> set[int]:
        tokens: set[int] = set()

        for token_text, token_id in self.vocab.items():
            if token_text.strip() in {",", "}"}:
                tokens.add(token_id)

        return tokens

    def _get_number_tokens(self) -> set[int]:
        tokens: set[int] = set()

        for token_text, token_id in self.vocab.items():
            if token_text.isdigit():
                tokens.add(token_id)

        return tokens

    def _get_string_tokens(self) -> set[int]:
        tokens: set[int] = set()

        for token_text, token_id in self.vocab.items():
            if '"' not in token_text and '\\' not in token_text:
                tokens.add(token_id)

        return tokens

    def _generate_number(self,
                         input_ids: list[int],
                         value_start: int) -> list[int]:

        value_tokens: list[int] = []

        number_tokens: set[int] = self._get_number_tokens()
        end_tokens: set[int] = self._get_number_end_tokens()

        while True:
            allowed_tokens: set[int] = number_tokens.copy()

            if value_tokens:
                allowed_tokens.update(end_tokens)

            next_token: int = self._get_next_token(
                input_ids + value_tokens,
                allowed_tokens
            )

            if next_token in end_tokens:
                break

            value_tokens.append(next_token)

        return value_tokens

    def _generate_string(
        self,
        input_ids: list[int],
        value_start: int,
        max_tokens: int
    ) -> list[int]:
        value_tokens: list[int] = []

        quote_token: int = self.vocab['"']
        value_tokens.append(quote_token)

        for _ in range(max_tokens):
            content_tokens: set[int] = self._get_string_tokens()

            logits: list[float] = self.llm.get_logits_from_input_ids(
                input_ids + value_tokens
            )
            logits_array = np.array(logits)

            quote_logit: float = float(logits_array[quote_token])

            best_content_token: int = max(
                content_tokens,
                key=lambda token_id: logits_array[token_id]
            )

            best_content_logit: float = float(
                logits_array[best_content_token]
            )

            print(
                "CURRENT:",
                repr(
                    self.llm.decode(
                        input_ids[value_start:] + value_tokens
                    )
                )
            )
            print("QUOTE:", quote_logit)
            print(
                "BEST CONTENT:",
                repr(self.llm.decode(best_content_token)),
                best_content_logit
            )

            allowed_tokens: set[int] = content_tokens | {quote_token}

            next_token: int = max(
                allowed_tokens,
                key=lambda token_id: logits_array[token_id]
            )

            value_tokens.append(next_token)

            print("TOKEN:", repr(self.llm.decode(next_token)))

            if next_token == quote_token:
                return value_tokens

        raise ValueError("String did not close within max_tokens")

    # def _generate_boolean(self,
    #                       input_ids: list[int],
    #                       value_start: int) -> list[int]:
    #     pass

    def _generate_value(self,
                        param_type: str,
                        input_ids: list[int],
                        value_start: int,
                        max_tokens: int) -> list[int]:
        if param_type == "number":
            return self._generate_number(input_ids, value_start)
        elif param_type == "string":
            return self._generate_string(input_ids, value_start, max_tokens)
        # elif param_type == "boolean":
        #     return self._generate_boolean(input_ids, value_start)

        raise ValueError(f"Unsupported parameter type: {param_type}")

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
                 max_tokens: int = 20) -> str:

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

            if self._has_finished_path(active_paths):
                state = States.PARAM_KEY

        current_fixed_path = self.param_key.copy()
        while state == States.PARAM_KEY:
            input_ids.append(current_fixed_path[0])
            current_fixed_path = current_fixed_path[1:]

            if not current_fixed_path:
                state = States.PARAM_NAME

        if state == States.PARAM_NAME:
            selected_funct: FunctionDefinition = self.functions[
                self._selected_func(active_paths)]

            total_parameters: int = len(selected_funct.parameters)
            for index, (param_name, param_schema) in enumerate(
                    selected_funct.parameters.items()):
                param_name_tokens: list[int] = self.llm.encode(
                    f'"{param_name}": ')[0].tolist()

                input_ids.extend(param_name_tokens)

                param_type: str = param_schema["type"]
                value_start: int = len(input_ids)

                value_tokens: list[int] = self._generate_value(
                    param_type,
                    input_ids,
                    value_start,
                    max_tokens
                )

                input_ids.extend(value_tokens)

                if index < total_parameters - 1:
                    input_ids.extend(
                        self.llm.encode(", ")[0].tolist()
                    )
                else:
                    input_ids.extend(
                        self.llm.encode("}")[0].tolist()
                    )

        return self.llm.decode(input_ids)
