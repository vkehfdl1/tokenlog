from abc import abstractmethod, ABC
from typing import List, Union, Dict

import tiktoken
from transformers import AutoTokenizer


Prompt = Union[str, List[Dict[str, str]]]


class BaseTokenizer(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = self._load_tokenizer()

    @abstractmethod
    def _load_tokenizer(self):
        pass

    @abstractmethod
    def get_tokens(self, text: Prompt) -> List[int]:
        pass

    @abstractmethod
    def get_tokens_batch(self, texts: List[Prompt]) -> List[List[int]]:
        pass


class TiktokenTokenizer(BaseTokenizer):
    def _load_tokenizer(self):
        return tiktoken.encoding_for_model(self.model_name)

    def get_tokens(self, text: Prompt) -> List[int]:
        return self.tokenizer.encode(self.convert_prompt(text))

    def get_tokens_batch(self, texts: List[Prompt]) -> List[List[int]]:
        parsed_strings = [self.convert_prompt(text) for text in texts]
        return self.tokenizer.encode_batch(parsed_strings, allowed_special='all')

    def convert_prompt(self, text: Prompt) -> str:
        if isinstance(text, str):
            return text
        elif isinstance(text, list):
            return self.messages_to_string(text)
        else:
            raise ValueError(f"Unsupported prompt format : {type(text)}")

    def messages_to_string(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to string format for accurate token counting"""
        formatted_parts = [
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
            for message in messages
        ]
        formatted_parts.append("<|im_start|>assistant")
        full_string = "\n".join(formatted_parts)
        return full_string


class HuggingfaceTokenizer(BaseTokenizer):

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def get_tokens(self, text: Prompt):
        return self.tokenizer([self.convert_prompt(text)]).data['input_ids'][0]

    def get_tokens_batch(self, texts: List[Prompt]):
        parsed_strings = [self.convert_prompt(text) for text in texts]
        return self.tokenizer(parsed_strings).data['input_ids']

    def convert_prompt(self, text: Prompt) -> str:
        if isinstance(text, str):
            return text
        elif isinstance(text, list):
            return self.tokenizer.apply_chat_template(text, tokenize=False)
        else:
            raise ValueError(f"Unsupported prompt format : {type(text)}")
