from flask import Flask, request, render_template
import pandas as pd
import re
import google.generativeai as palm
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize Google PaLM API
palm.configure(api_key=os.getenv("PALM_API_KEY"))

# Load the dataset
df = pd.read_csv('/mnt/data/IMDB Dataset.csv')

# Clean the text data
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

# Function to get the text from PaLM model
def extract_features_with_palm(text):
    try:
        response = palm.generate_text(prompt=text, temperature=0.0)
        if response.filters:
            print(f"Blocked reason: {response.filters[0]['reason']}")
            return None
        if response.candidates:
            return response.candidates[0]['output'].strip()
        else:
            return None
    except Exception as e:
        print(f"Error with PaLM model: {e}")
        return None

# Prepare the data
df['palm_output'] = df['cleaned_review'].apply(extract_features_with_palm)
df = df.dropna(subset=['palm_output'])

# Train a simple model
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(df['palm_output'])
y = df['sentiment']

classifier = LogisticRegression()
classifier.fit(X_vec, y)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned_review = clean_text(review)
    palm_output = extract_features_with_palm(cleaned_review)
    
    if palm_output is None:
        return "Sorry, we couldn't process your request."

    palm_vec = vectorizer.transform([palm_output])
    prediction = classifier.predict(palm_vec)[0]
    
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return render_template('index.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
