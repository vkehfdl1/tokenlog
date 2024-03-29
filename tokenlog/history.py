import uuid
from dataclasses import dataclass, field
from typing import Optional, List

from datetime import datetime


@dataclass
class History:
    token_length: int
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    text: Optional[str] = None
    tokens: Optional[List[int]] = None
    answer: list = field(default_factory=list)
    log_datetime: datetime = field(default_factory=datetime.now)
