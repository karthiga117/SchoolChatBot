from typing import List


def chunk_text(text: str, min_size: int = 500, max_size: int = 1000) -> List[str]:
    """Split a large text into chunks between ``min_size`` and ``max_size`` characters.

    This implementation splits on whitespace and then merges adjacent chunks if
    they fall below the minimum size threshold.  It does not attempt to preserve
    sentence boundaries; for a production system you might integrate a proper
    tokenizer or use NLP libraries.
    """
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    for word in words:
        current.append(word)
        if len(" ".join(current)) >= max_size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))

    # ensure minimum size by merging small trailing segments
    merged: List[str] = []
    for chunk in chunks:
        if merged and len(merged[-1]) < min_size:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged
