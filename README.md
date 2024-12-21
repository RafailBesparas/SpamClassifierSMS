# SpamClassifierSMS
This project implements a Spam Message Classifier using Flask to deploy a machine learning model that predicts whether a message is spam or not spam. The classifier uses a Bernoulli Naive Bayes model (BernouliClassifier.pkl) and a CountVectorizer (vectorizer.pkl) to preprocess and classify user-inputted messages.

# How the Project Works:
1. Text Preprocessing:
The app uses a transform_text function to preprocess user input by:
Converting text to lowercase.
Tokenizing (splitting) the text into words.
Removing non-alphanumeric characters, stopwords, and punctuation.
Stemming (reducing words to their root form).

2. Web Interface:
A simple HTML form (index.html) allows users to input a message.
After submission, the text is passed to the Flask app for prediction.
Prediction Process:

The input is vectorized using the CountVectorizer (loaded from vectorizer.pkl).
The pre-trained Bernoulli Naive Bayes model classifies the message.
The app returns a prediction – either Spam or Not Spam.
Routes:

3.  Displays the HTML form for message input.
/predict – Handles POST requests to process and classify the message.


