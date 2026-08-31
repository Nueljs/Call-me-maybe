from enum import Enum


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
        self.token_texts: dict[int, str] = {
            token_id: token_text
            for token_text, token_id in vocab.items()
        }

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