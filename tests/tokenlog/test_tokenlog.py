import uuid

import pytest

import tokenlog


@pytest.fixture
def token_logger():
    logger = tokenlog.getLogger('test', 'gpt-3.5-turbo')
    yield logger
    logger.clear()


def test_singleton():
    log1 = tokenlog.getLogger('test1', 'gpt-3.5-turbo')
    log2 = tokenlog.getLogger('test2', 'gpt-3.5-turbo')
    log3 = tokenlog.getLogger('test1', 'gpt-3.5-turbo')

    assert log1 is log3
    assert log1 is not log2


def test_get_history(token_logger):
    token_logger.query('This is a test text.')
    token_logger.query('This is another test text.')
    token_logger.query('This is a third test text.')

    history = token_logger.get_history()

    assert len(history) == 3
    assert all([isinstance(h, tokenlog.History) for h in history])
    assert history[0].text == 'This is a test text.'
    assert history[1].text == 'This is another test text.'
    assert history[2].text == 'This is a third test text.'

    # test clear method
    token_logger.clear()

    assert len(token_logger.get_history()) == 0


def test_get_token_usage(token_logger):
    q1 = token_logger.query('This is a test text.')
    token_logger.query('This is another test text.')
    token_logger.query('This is a third test text.')
    token_logger.answer('This is the answer of first query.', q1)

    assert token_logger.get_token_usage() == 19
    token_logger.query('This is a fourth test text.')
    assert token_logger.get_token_usage() > 19
    token_logger.clear()
    assert token_logger.get_token_usage() == 0


def test_query_answer(token_logger):
    q1 = token_logger.query('This is a test text.')
    q2 = token_logger.query('This is another test text.')
    token_logger.answer('This is the answer of first query.', q1)
    q3 = token_logger.query([
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, how are you?"},
    ])
    token_logger.answer('This is the second answer of first query.', q1)
    token_logger.answer('This is the answer of second query.', q2)
    token_logger.answer([
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    ], q3)
    assert q2 != q1
    assert isinstance(q2, uuid.UUID)
    assert isinstance(q1, uuid.UUID)

    histories = token_logger.get_history()
    assert len(histories) == 3
    assert len(histories[0].answer) == 2
    assert len(histories[1].answer) == 1
    assert len(histories[2].answer) == 1


def test_query_batch(token_logger):
    q_ids = token_logger.query_batch(['This is a test text.',
                                      'This is another test text.',
                                      'This is a third test text.'])
    token_logger.answer_batch(['This is the answer of first query.',
                               'This is the answer of second query.',
                               'Thius is the answer of first query 2.'], [q_ids[0], q_ids[1], q_ids[0]])
    assert len(q_ids) == 3
    histories = token_logger.get_history()
    assert len(histories) == 3
    assert len(histories[0].answer) == 2
    assert len(histories[1].answer) == 1
    assert len(histories[2].answer) == 0
