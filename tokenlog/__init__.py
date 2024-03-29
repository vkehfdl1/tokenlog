import uuid
from typing import Optional, List
from dataclasses import dataclass, field

from datetime import datetime
import tiktoken.model

from tokenlog.tokenizer import TiktokenTokenizer, HuggingfaceTokenizer

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

_nameToLevel = {
    'CRITICAL': CRITICAL,
    'FATAL': FATAL,
    'ERROR': ERROR,
    'WARN': WARNING,
    'WARNING': WARNING,
    'INFO': INFO,
    'DEBUG': DEBUG,
    'NOTSET': NOTSET,
}

GPT_MODEL_NAMES = tiktoken.model.MODEL_TO_ENCODING.keys()

import logging

logging.getLogger()


def get_tokenizer(model_name: str):
    if model_name in GPT_MODEL_NAMES:
        return TiktokenTokenizer(model_name)
    else:
        return HuggingfaceTokenizer(model_name)


def getLogger(name: str, model_name: Optional[str] = None):
    if not name or isinstance(name, str):
        raise ValueError('name must be a string')
    return TokenLogger(name, model_name)


class TokenLogger:
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
