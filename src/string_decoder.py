from enum import Enum


class StringState(Enum):
    CONTENT = 1
    ESCAPE = 2
    UNICODE_1 = 3
    UNICODE_2 = 4
    UNICODE_3 = 5
    UNICODE_4 = 6
    CLOSED = 7