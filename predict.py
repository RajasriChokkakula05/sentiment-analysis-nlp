import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load saved model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

print("Sentiment Analyzer Ready 🚀")

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