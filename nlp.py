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

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

