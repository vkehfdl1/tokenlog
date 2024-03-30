# tokenlog

Simplest token log system for your LLM, embedding model calls.

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

t_logger = tokenlog.getLogger('session_1', 'gpt-3.5-turbo') # write logger name and model name that you are using
q1 = t_logger.query('This is the query that you used in LLM') # log the query

t_logger.answer('This is an answer from LLM', q1) # log the answer

t_logger.get_token_usage() # get total token usage from all queries

t_logger.get_history() # get history of token usage

t_logger.clear() # clear all histories
```


## Support Models

We support all **OpenAI** models with tiktoken and **Huggingface** models that support `AutoTokenizer`.


## Use Case

This library used in [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) project.


## To-do

- [ ] Add Handlers for exporting logs
- [ ] Support more models
- [ ] Batch logging
