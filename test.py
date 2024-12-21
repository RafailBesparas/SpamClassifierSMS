from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load('BernouliClassifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # Assuming CountVectorizer

ps = PorterStemmer()

# Preprocess function (reuse your transformation_text)
def transform_text(text):
    text = text.lower()
    text = text.split()

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Route to render HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    transformed_message = transform_text(message)
    vectorized_message = vectorizer.transform([transformed_message])
    
    prediction = model.predict(vectorized_message)[0]
    
    result = 'Spam' if prediction == 1 else 'Not Spam'
    return render_template('index.html', prediction_text=f'This message is: {result}')

if __name__ == '__main__':
    app.run(debug=True)
