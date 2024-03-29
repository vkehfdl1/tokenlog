import tokenlog


def test_singleton():
    log1 = tokenlog.getLogger('test1', 'gpt-3.5-turbo')
    log2 = tokenlog.getLogger('test2', 'gpt-3.5-turbo')
    log3 = tokenlog.getLogger('test1', 'gpt-3.5-turbo')

    assert log1 is log3
    assert log1 is not log2
