import uuid
from typing import Optional, List
from dataclasses import dataclass, field

from datetime import datetime
import tiktoken.model

from tokenlog.tokenizer import TiktokenTokenizer, HuggingfaceTokenizer

GPT_MODEL_NAMES = tiktoken.model.MODEL_TO_ENCODING.keys()


def get_tokenizer(model_name: str):
    if model_name in GPT_MODEL_NAMES:
        return TiktokenTokenizer(model_name)
    else:
        return HuggingfaceTokenizer(model_name)


def getLogger(name: str, model_name: Optional[str] = None):
    if not name or isinstance(name, str):
        raise ValueError('name must be a string')
    return TokenLogger(name, model_name)


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        name = kwargs.get('name', None) if kwargs else args[0]  # Assuming 'name' is the first argument
        if name not in cls._instances:
            cls._instances[name] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[name]


class TokenLogger(metaclass=SingletonMeta):
    def __init__(self, name: str, model: str):
        self.name = name
        self.tokenizer = get_tokenizer(model)
        self.history: List[History] = []

    def get_token_usage(self) -> int:
        return sum(list(map(lambda x: x.token_length, self.history)))

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []

    def __record_token_usage(self, text: str):
        tokens = self.tokenizer.get_tokens(text)
        history = History(
            token_length=len(tokens),
            text=text,
            tokens=tokens
        )
        self.history.append(history)
        return history

    def __find_history(self, _id: uuid.uuid4):
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i].id == _id:
                return self.history[i]
        return None

    def query(self, text: str):
        history = self.__record_token_usage(text)
        return history.id

    def answer(self, text: str, query_id: uuid.uuid4):
        history = self.__record_token_usage(text)
        query_history = self.__find_history(query_id)
        if query_history is None:
            raise ValueError(f'query_id not found. Your input query_id is {query_id}')
        query_history.answer.append(history)


@dataclass
class History:
    token_length: int
    id: uuid.uuid4 = field(default_factory=uuid.uuid4)
    text: Optional[str] = None
    tokens: Optional[List[int]] = None
    answer: List['History'] = field(default_factory=list)
    datetime: datetime = field(default_factory=datetime.now)
