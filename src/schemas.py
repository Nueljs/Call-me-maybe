from pydantic import BaseModel


class ReverseString(BaseModel):
    def __init__(self, s: str) -> None:
        self.s: str = s