import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re

def ensure_nltk_data():
    resources = ['punkt_tab', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt_tab' else f'corpora/{resource}')
        except LookupError:
            print(f"Descargando NLTK: {resource}...")
            nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()

def clean_sentence(sentence: str):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]

def get_synonyms(word: str):
    return [l.name() for syn in wordnet.synsets(word) for l in syn.lemmas()]

def extract_order_number(text: str) -> str | None:
    text = text.lower().strip()
    patterns = [
        r'(?:order|it\'s|my order is|the order is)[\s#:]*([a-z0-9]{5,})',
        r'#\s*([a-z0-9]{5,})',
        r'\b([a-z0-9]{5,})\b'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).upper()
            if candidate.lower() not in {'order', 'number', 'code'}:
                return candidate

    return None