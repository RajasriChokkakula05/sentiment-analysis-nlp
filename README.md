# Sentiment Analysis using NLP

This project performs **Sentiment Analysis** on text such as tweets or product reviews using **Natural Language Processing (NLP)** and **Machine Learning**.

The model analyzes text and classifies it as **Positive 😀 or Negative 😡**.

---

## Project Overview

Sentiment analysis is a common NLP task used to understand people's opinions from text data such as reviews, tweets, or comments.

In this project:

- Text data is cleaned using NLP techniques
- Stopwords are removed
- Text is converted to numerical features using **TF-IDF**
- A **Naive Bayes classifier** is trained to classify sentiment

---

## Technologies Used

- Python
- NLTK
- Scikit-learn
- Pandas
- NumPy
- TF-IDF Vectorization
- Naive Bayes Classifier

---

## Project Structure
sentiment-analysis/
│
├── sentiment_analysis.py # Model training
├── predict.py # Prediction script
├── sentiment_model.pkl # Saved ML model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## How to Run the Project

### 1 Install dependencies

pip install -r requirements.txt


### 2 Train the model

python sentiment_analysis.py


### 3 Run prediction

python predict.py


---

## Example

Input:

This movie is amazing


Output:

Positive 😀


Input:
Worst product ever

Output:
Negative 😡


---

## Internship Task

This project was developed as **Task 3 of the AI Internship at Kodbud**.

---

## Author

**Raji**

GitHub:  
https://github.com/RajasriChokkakula05