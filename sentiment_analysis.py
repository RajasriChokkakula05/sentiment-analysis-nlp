import nltk
import pandas as pd
import re
import pickle

from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download required datasets
nltk.download('movie_reviews')
nltk.download('stopwords')

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        review = movie_reviews.raw(fileid)
        documents.append((review, category))

data = pd.DataFrame(documents, columns=["review", "sentiment"])

print("Dataset Sample:")
print(data.head())

# -----------------------------
# STEP 2: Convert Labels
# -----------------------------

data['sentiment'] = data['sentiment'].map({
    'pos': 1,
    'neg': 0
})

# -----------------------------
# STEP 3: Text Cleaning (NLP)
# -----------------------------

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

data['clean_text'] = data['review'].apply(clean_text)

# -----------------------------
# STEP 4: Prepare Features
# -----------------------------

X = data['clean_text']
y = data['sentiment']

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 6: Convert Text → Numbers
# -----------------------------

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# STEP 7: Train Model
# -----------------------------

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save trained model
pickle.dump(model, open("sentiment_model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and Vectorizer saved successfully!")

# -----------------------------
# STEP 8: Evaluate Model
# -----------------------------

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# -----------------------------
# STEP 9: Predict Custom Input
# -----------------------------

while True:

    text = input("\nEnter a review (or type exit): ")

    if text.lower() == "exit":
        break

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("Positive 😀")
    else:
        print("Negative 😡")