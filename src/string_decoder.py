from enum import Enum
from llm_sdk import Small_LLM_Model  # type: ignore


class StringState(Enum):
    CONTENT = 1
    ESCAPE = 2
    UNICODE_1 = 3
    UNICODE_2 = 4
    UNICODE_3 = 5
    UNICODE_4 = 6
    CLOSED = 7


class StringDecoder:
    """Handle constrained JSON string generation."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab: dict[str, int] = vocab
        self.quote_token: int = vocab['"']

        self.token_texts: dict[int, str] = {
            token_id: token_text
            for token_text, token_id in vocab.items()
        }

        self.allowed_tokens: dict[StringState, set[int]] = {}

        for state in StringState:
            if state != StringState.CLOSED:
                self.allowed_tokens[state] = self._build_allowed_tokens(state)

    def _advance_state(self,
                       token_text: str,
                       state: StringState) -> StringState | None:
        """Return the state reached after reading a token."""
        for char in token_text:
            if state == StringState.CONTENT:
                if char == '"':
                    state = StringState.CLOSED
                elif char == '\\':
                    state = StringState.ESCAPE

            elif state == StringState.ESCAPE:
                if char in '"\\/bfnrt':
                    state = StringState.CONTENT
                elif char == 'u':
                    state = StringState.UNICODE_1
                else:
                    return None

            elif state == StringState.UNICODE_1:
                if char not in "0123456789abcdefABCDEF":
                    return None
                state = StringState.UNICODE_2

            elif state == StringState.UNICODE_2:
                if char not in "0123456789abcdefABCDEF":
                    return None
                state = StringState.UNICODE_3

            elif state == StringState.UNICODE_3:
                if char not in "0123456789abcdefABCDEF":
                    return None
                state = StringState.UNICODE_4

            elif state == StringState.UNICODE_4:
                if char not in "0123456789abcdefABCDEF":
                    return None
                state = StringState.CONTENT

            elif state == StringState.CLOSED:
                return None

        return state

    def _build_allowed_tokens(
        self,
        state: StringState,
    ) -> set[int]:
        """Build the token IDs valid from the current state."""
        allowed_tokens: set[int] = set()

        for token_id, token_text in self.token_texts.items():
            next_state = self._advance_state(
                token_text,
                state
            )

            if next_state is not None:
                allowed_tokens.add(token_id)

        return allowed_tokens

    def _get_allowed_tokens(
        self,
        state: StringState,
    ) -> set[int]:
        """Return precomputed allowed token IDs."""
        return self.allowed_tokens[state]

    def generate(
        self,
        llm: Small_LLM_Model,
        input_ids: list[int],
        max_tokens: int,
    ) -> list[int]:
        """Generate a constrained JSON string."""
        value_tokens: list[int] = []

        quote_token: int = self.quote_token
        value_tokens.append(quote_token)

        state: StringState = StringState.CONTENT

        for token_index in range(max_tokens):
            if token_index == max_tokens - 1:
                allowed_tokens: set[int] = {quote_token}
            else:
                allowed_tokens = self._get_allowed_tokens(state)

            logits: list[float] = llm.get_logits_from_input_ids(
                input_ids + value_tokens
            )

            next_token: int = max(
                allowed_tokens,
                key=lambda token_id: logits[token_id]
            )

            value_tokens.append(next_token)

            token_text: str = self.token_texts[next_token]

            next_state: StringState | None = self._advance_state(
                token_text,
                state,
            )

            if next_state is None:
                raise ValueError(
                    "Generated token is invalid for current string state"
                )

            state = next_state

            if state == StringState.CLOSED:
                return value_tokens

        raise ValueError("JSON can't be completed with that number of tokens")
