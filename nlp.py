import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

data = pd.read_csv('UpdatedResumeDataSet.csv')


def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.lower()


data['cleaned_resume'] = data['Resume'].apply(clean_text)

#nltk.data.find('stopword')
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return words

data['tokenized_text'] = data['cleaned_resume'].apply(tokenize_and_remove_stopwords)

print(data['tokenized_text'])