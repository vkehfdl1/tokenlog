# tokenlog

Python logger that logs token usage.
Looks like Python logger, easy to use. 
Check token usage easily in the code base.

## Installation
Get on pypi.

```bash
pip install tokenlog
```

## How to use

Start with initializing the logger.
Each logger with the same name is singleton.

```python
import tokenlog

t_logger = tokenlog.getLogger('session_1',
                              model='gpt-3.5-turbo')
t_logger.debug('model input lengths')

t_logger.get_token_usage() # get total token usage

t_logger.get_history() # get history of token usage

tokenlog.all_clear() # clear all token usage in all loggers
```
