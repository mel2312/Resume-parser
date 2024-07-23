import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

#train
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, data['Category'], test_size=0.2, random_state=42)

print(data['Category'].unique())

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

#check accuracy
print("\n\n\n", classification_report(y_test,predictions))
print("\n\n\n", confusion_matrix(y_test,predictions))

error_rate = []
for i in range(1,25):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i= knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,25),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='green',markersize=10)
plt.title("Error Rate vs K-value")
plt.xlabel("K-value")
plt.ylabel("Error Rate")
plt.show()

print("\n\n\n Best value are of K is from 1 to 3")