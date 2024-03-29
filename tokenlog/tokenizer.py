from abc import abstractmethod, ABC
from typing import List

import tiktoken
from transformers import AutoTokenizer


class BaseTokenizer(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = self._load_tokenizer()

    @abstractmethod
    def _load_tokenizer(self):
        pass

    @abstractmethod
    def get_tokens(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def get_tokens_batch(self, texts: List[str]) -> List[List[int]]:
        pass


class TiktokenTokenizer(BaseTokenizer):
    def _load_tokenizer(self):
        return tiktoken.encoding_for_model(self.model_name)

    def get_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def get_tokens_batch(self, texts: List[str]) -> List[List[int]]:
        return self.tokenizer.encode_batch(texts, allowed_special='all')


class HuggingfaceTokenizer(BaseTokenizer):

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def get_tokens(self, text: str):
        return self.tokenizer([text]).data['input_ids'][0]

    def get_tokens_batch(self, texts: List[str]):
        return self.tokenizer(texts).data['input_ids']
