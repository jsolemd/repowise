"""Search intelligence module for doc-search.

This module provides query understanding and intent detection to optimize
search strategies for different types of documentation queries.

## Query Intent Types

| Intent | Detection Pattern | Search Strategy |
|--------|-------------------|-----------------|
| API_LOOKUP | CamelCase, function names | BM25-heavy, reference docs |
| HOW_TO | "how to", "how do I" | Semantic-heavy, guide docs |
| ERROR_DEBUG | "error", "fix", "mismatch" | Semantic + troubleshooting |
| CONCEPT | General terms | Balanced hybrid |

## Usage

```python
from repowise.docs.search.intent import detect_intent, QueryIntent

intent = detect_intent("useEffect cleanup")
# Returns QueryIntent.API_LOOKUP

intent = detect_intent("how to handle authentication")
# Returns QueryIntent.HOW_TO
```
"""

from repowise.docs.search.intent import (
    INTENT_CATEGORY_BOOSTS,
    INTENT_SEARCH_WEIGHTS,
    QueryIntent,
    detect_intent,
)

__all__ = [
    "INTENT_CATEGORY_BOOSTS",
    "INTENT_SEARCH_WEIGHTS",
    "QueryIntent",
    "detect_intent",
]
