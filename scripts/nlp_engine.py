import re
from collections import Counter

POSITIVE_WORDS = {
    "good", "great", "positive", "growth", "gain", "up", "bull", "profit",
    "strong", "better", "improve", "excellent", "rise", "increase"
}

NEGATIVE_WORDS = {
    "bad", "poor", "negative", "loss", "down", "bear", "risk", "weak",
    "decline", "drop", "worse", "fall", "decrease"
}

STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "are",
    "was", "were", "will", "been", "into", "your", "you", "our", "their"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return [w for w in clean_text(text).split() if len(w) > 2]

def get_sentiment(text):
    tokens = tokenize(text)
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)

    if pos > neg:
        sentiment = "Positive"
    elif neg > pos:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "clean_text": clean_text(text),
        "tokens": tokens,
        "sentiment": sentiment,
        "positive_score": pos,
        "negative_score": neg
    }

def get_top_keywords(text, top_n=5):
    tokens = tokenize(text)
    words = [w for w in tokens if w not in STOP_WORDS]
    return [{"keyword": k, "count": int(v)} for k, v in Counter(words).most_common(top_n)]