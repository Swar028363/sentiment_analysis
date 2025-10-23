def get_positive_words(words: list[str], positive_words: set[str] = set()) -> list[str]:
    """Get all the Positive words form the words list. Positive words are like good, great, love, etc."""
    return [w for w in words if w in positive_words]

def get_negative_words(words: list[str], negative_words: set[str] = set()) -> list[str]:
    """Get all the Negative words from the words list. Negative words are like bad, worst, hate, etc."""
    return [w for w in words if w in negative_words]