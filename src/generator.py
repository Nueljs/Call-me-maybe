import json
from enum import Enum

import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore
from .schemas import FunctionDefinition

from .string_decoder import StringDecoder


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

    def __init__(
        self,
        llm: Small_LLM_Model,
        functions: list[FunctionDefinition],
    ) -> None:
        """Initialize the Generator with the LLM and target functions."""
        self.llm: Small_LLM_Model = llm
        self.vocab: dict[str, int] = self._load_vocab()
        self.string_decoder: StringDecoder = StringDecoder(self.vocab)
        self.functions: list[FunctionDefinition] = functions

        self.functions_names: list[str] = [
            f'"{function.name}",' for function in self.functions
        ]

        self.fn_token_paths: list[tuple[int, list[int]]] = []

        for index, fn_name in enumerate(self.functions_names):
            tokens = self.llm.encode(fn_name)[0].tolist()
            self.fn_token_paths.append((index, tokens))

        self.name_key: list[int] = self.llm.encode(
            '\n "name": '
        )[0].tolist()

        self.param_key: list[int] = self.llm.encode(
            '\n "parameters": {'
        )[0].tolist()

    def _load_vocab(self) -> dict[str, int]:
        """Load the model vocabulary."""
        path_vocab: str = self.llm.get_path_to_vocab_file()

        with open(path_vocab, "r", encoding="utf-8") as f:
            data: dict[str, int] = json.load(f)

        return data

    def _selected_func(
        self,
        active_paths: list[tuple[int, list[int]]],
    ) -> int:
        """Return the index of the selected function."""
        for func in active_paths:
            if not func[1]:
                return func[0]

        raise ValueError("No function path has finished")

    def _advance_func_paths(
        self,
        active_paths: list[tuple[int, list[int]]],
        token: int,
    ) -> list[tuple[int, list[int]]]:
        """Advance function paths using the generated token."""
        new_active_paths: list[tuple[int, list[int]]] = []

        for path in active_paths:
            if path[1] and path[1][0] == token:
                new_active_paths.append(
                    (path[0], path[1][1:])
                )

        return new_active_paths

    def _get_allowed_tokens(
        self,
        state: States,
        active_paths: list[tuple[int, list[int]]] | None = None,
    ) -> set[int]:
        """Return allowed tokens for the current state."""
        allowed_tokens: set[int] = set()

        if state == States.START:
            allowed_tokens.add(self.vocab["{"])

        elif state == States.FUNC_NAME:
            if active_paths is None:
                raise ValueError(
                    "active_paths required in FUNC_NAME"
                )

            for path in active_paths:
                if path[1]:
                    allowed_tokens.add(path[1][0])

        return allowed_tokens

    def _has_finished_path(
        self,
        active_paths: list[tuple[int, list[int]]],
    ) -> bool:
        """Check whether one function path has finished."""
        return any(not path[1] for path in active_paths)

    def _get_number_end_tokens(self) -> set[int]:
        """Return tokens that can terminate a number."""
        tokens: set[int] = set()

        for token_text, token_id in self.vocab.items():
            if token_text.strip() in {",", "}"}:
                tokens.add(token_id)

        return tokens

    def _get_number_tokens(self) -> set[int]:
        """Return numeric tokens from the vocabulary."""
        tokens: set[int] = set()

        for token_text, token_id in self.vocab.items():
            if token_text.isdigit():
                tokens.add(token_id)

        return tokens

    def _generate_number(
        self,
        input_ids: list[int],
        value_start: int,
    ) -> list[int]:
        """Generate a numeric value."""
        value_tokens: list[int] = []

        number_tokens: set[int] = self._get_number_tokens()
        end_tokens: set[int] = self._get_number_end_tokens()

        while True:
            allowed_tokens: set[int] = number_tokens.copy()

            if value_tokens:
                allowed_tokens.update(end_tokens)

            next_token: int = self._get_next_token(
                input_ids + value_tokens,
                allowed_tokens,
            )

            if next_token in end_tokens:
                break

            value_tokens.append(next_token)

        return value_tokens

    def _generate_value(
        self,
        param_type: str,
        input_ids: list[int],
        value_start: int,
        max_tokens: int,
    ) -> list[int]:
        """Generate a value according to its declared type."""
        if param_type == "number":
            return self._generate_number(
                input_ids,
                value_start,
            )

        if param_type == "string":
            return self.string_decoder.generate(
                self.llm,
                input_ids,
                max_tokens,
            )

        raise ValueError(
            f"Unsupported parameter type: {param_type}"
        )

    def _get_next_token(
        self,
        input_ids: list[int],
        allowed_tokens: set[int],
    ) -> int:
        """Select the highest-logit token from the allowed set."""
        logits: list[float] = (
            self.llm.get_logits_from_input_ids(input_ids)
        )

        logits_array = np.array(logits)
        mask = np.full_like(logits_array, -np.inf)

        for token_id in allowed_tokens:
            mask[token_id] = logits_array[token_id]

        return int(np.argmax(mask))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 40,
    ) -> str:
        """Generate a constrained function call."""
        state: States = States.START

        input_ids: list[int] = (
            self.llm.encode(prompt)[0].tolist()
        )

        generation_start: int = len(input_ids)

        active_paths: list[tuple[int, list[int]]] = [
            (index, path.copy())
            for index, path in self.fn_token_paths
        ]

        selected_funct: FunctionDefinition | None = None
        parameters: list[tuple[str, dict[str, str]]] = []
        current_param_index: int = 0

        while state != States.END:

            if state == States.START:
                allowed_tokens: set[int] = (
                    self._get_allowed_tokens(state)
                )

                next_token: int = self._get_next_token(
                    input_ids,
                    allowed_tokens,
                )

                input_ids.append(next_token)

                state = States.NAME_KEY

            elif state == States.NAME_KEY:
                input_ids.extend(self.name_key)

                state = States.FUNC_NAME

            elif state == States.FUNC_NAME:
                allowed_tokens = self._get_allowed_tokens(
                    state,
                    active_paths,
                )

                next_token = self._get_next_token(
                    input_ids,
                    allowed_tokens,
                )

                input_ids.append(next_token)

                active_paths = self._advance_func_paths(
                    active_paths,
                    next_token,
                )

                if self._has_finished_path(active_paths):
                    selected_funct = self.functions[
                        self._selected_func(active_paths)
                    ]

                    parameters = list(
                        selected_funct.parameters.items()
                    )

                    state = States.PARAM_KEY

            elif state == States.PARAM_KEY:
                input_ids.extend(self.param_key)

                state = States.PARAM_NAME

            elif state == States.PARAM_NAME:
                param_name, _ = parameters[
                    current_param_index
                ]

                param_name_tokens: list[int] = (
                    self.llm.encode(
                        f'"{param_name}": '
                    )[0].tolist()
                )

                input_ids.extend(param_name_tokens)

                state = States.PARAM_VALUE

            elif state == States.PARAM_VALUE:
                param_name, param_schema = parameters[
                    current_param_index
                ]

                param_type: str = param_schema["type"]

                value_start: int = len(input_ids)

                value_tokens: list[int] = (
                    self._generate_value(
                        param_type,
                        input_ids,
                        value_start,
                        max_tokens,
                    )
                )

                input_ids.extend(value_tokens)

                current_param_index += 1

                if current_param_index < len(parameters):
                    input_ids.extend(
                        self.llm.encode(", ")[0].tolist()
                    )

                    state = States.PARAM_NAME

                else:
                    state = States.END

        input_ids.extend(
            self.llm.encode("}}")[0].tolist()
        )

        return self.llm.decode(input_ids[generation_start:])
