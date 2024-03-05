import os
import ssl
import json
import pickle
import nltk
import numpy as np
import urllib3
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
import random

# Suppress SSL warnings
ssl._create_default_https_context = ssl._create_unverified_context

# Disable SSL warnings in urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load the pre-trained model and other necessary files
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())

# Use pickle.load instead of np.load
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

lemmatizer = nltk.WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    # Ensure the bag has the correct number of features (9 in this case)
    return np.array(bag)[:9]


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    error_threshold = 0.5  # Adjust this threshold as needed
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# ... (previous code remains unchanged)

# ... (previous code remains unchanged)

def get_response(intents, intents_json):
    print("intents: ", intents)
    print(intents_json)
    if intents:
        # Sort intents by probability in descending order
        sorted_intents = sorted(intents, key=lambda x: float(x['probability']), reverse=True)
        # Get the tag of the intent with the highest probability
        tag = sorted_intents[0]['intent']
        # Find the corresponding response in intents_json
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                result = random.choice(intent['responses'])
                return result

    # If no intent is found, return a default response
    return random.choice(intents_json['intents'][0]['responses'])




# ... (rest of the code remains unchanged)


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    user_msg = request.args.get('msg')
    response = chatbot_response(user_msg)
    return str(response)

if __name__ == '__main__':
    app.run(debug=True)
