from json import encoder

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('UpdatedResumeDataSet.csv')


def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.lower()


data['cleaned_resume'] = data['Resume'].apply(clean_text)
#subornodebbappon20@gmail.com
#nltk.data.find('stopword')
#nltk.download('stopwords')
#nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def tokenize_and_remove_stopwords(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words and word.isalpha()]
    return '.'.join(words)

data['tokenized_text'] = data['cleaned_resume'].apply(tokenize_and_remove_stopwords)
assert data['tokenized_text'].apply(lambda x: isinstance(x, str)).all(), "Not all processed resumes are strings"


tfidf_vectorizer = TfidfVectorizer(max_features=3000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['tokenized_text'])
print(data['Category'].value_counts())

#label encoding
labelencoder = LabelEncoder()

data['Category'] = labelencoder.fit_transform(data['Category'])

print(data.dtypes)
# Get the mapping dictionary
label_to_value = {label: idx for idx, label in enumerate(labelencoder.classes_)}


# Print the mapping

print("\n\nLabel to Value Mapping:")
for label, value in label_to_value.items():
    print(f"{label}: {value}")
