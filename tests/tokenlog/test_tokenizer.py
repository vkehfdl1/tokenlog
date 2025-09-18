from tokenlog.tokenizer import TiktokenTokenizer, HuggingfaceTokenizer

test_text = 'This is a test text.'
test_texts = [test_text] * 3

test_chat_message = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "What's the weather like?"}
]

test_chat_messages = [test_chat_message] * 3


def base_test_tokenizer(tokenizer):
    token = tokenizer.get_tokens(test_text)
    token_length = len(token)
    assert isinstance(token_length, int)
    assert token_length > 0

    tokens = tokenizer.get_tokens_batch(test_texts)
    token_lengths = list(map(len, tokens))
    assert isinstance(token_lengths, list)
    assert len(token_lengths) == len(test_texts)
    assert all(isinstance(length, int) for length in token_lengths)
    assert all(length > 0 for length in token_lengths)

def base_test_chat_tokenizer(tokenizer):
    token = tokenizer.get_tokens(test_chat_message)
    token_length = len(token)
    assert isinstance(token_length, int)
    assert token_length > 0

    tokens = tokenizer.get_tokens_batch(test_chat_messages)
    token_lengths = list(map(len, tokens))
    assert isinstance(token_lengths, list)
    assert len(token_lengths) == len(test_chat_messages)
    assert all(isinstance(length, int) for length in token_lengths)
    assert all(length > 0 for length in token_lengths)


def test_tiktoken_tokenizer():
    tiktoken_tokenizer = TiktokenTokenizer('gpt-3.5-turbo')
    base_test_tokenizer(tiktoken_tokenizer)


def test_tiktoken_chat_tokenizer():
    tiktoken_tokenizer = TiktokenTokenizer('gpt-3.5-turbo')
    base_test_chat_tokenizer(tiktoken_tokenizer)

def test_huggingface_tokenizer():
    huggingface_tokenizer = HuggingfaceTokenizer('facebook/opt-125m')
    base_test_tokenizer(huggingface_tokenizer)

def test_huggingface_chat_tokenizer():
    huggingface_tokenizer = HuggingfaceTokenizer('Qwen/Qwen3-8B')
    base_test_chat_tokenizer(huggingface_tokenizer)
