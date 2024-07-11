from nltk.data import find

try:
    find('corpora/stopwords')
    find('tokenizers/punkt')
    print("Stopwords and punkt data found!")
except LookupError:
    print("Stopwords or punkt data not found.")