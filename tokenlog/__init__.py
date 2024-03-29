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
#
# root = RootLogger(WARNING)
#
#
#
# def getLogger(name=None):
#     """
#     Return a logger with the specified name, creating it if necessary.
#
#     If no name is specified, return the root logger.
#     """
#     if not name or isinstance(name, str) and name == root.name:
#         return root
#     return Logger.manager.getLogger(name)


import logging

logging.getLogger()


def get_tokenizer(model_name: str):
    pass

class TokenLogger:
    def __init__(self, name: str, model_name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.tokenizer = get_tokenizer(model_name)

