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
    print("sentence_words: ",sentence_words)
    return sentence_words

def bow(sentence, words, show_details=True):
    print("words: ",words)
    x=len(words)
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    print("bag:",bag)
    print("np.array(bag)[:9] : ",np.array(bag)[:9])
    return np.array(bag)[:9]


def predict_class(sentence, model):
    #print("words/; ", words)
    p = bow(sentence, words, show_details=False)
    res1 = model.predict(np.array([p]))[0]
    print('res1: ', res1)
    error_threshold = 0.0  # Adjust this threshold as needed
    results = [[i, r] for i, r in enumerate(res1) if r > error_threshold]
    print('results: ', results)
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print("return List: ",return_list)
    return return_list


# ... (previous code remains unchanged)

# ... (previous code remains unchanged)

def get_response(intents, intents_json):
    #print("intents: ", intents)
    #print(intents_json)
    if intents:
        # Sort intents by probability in descending order
            #sorted_intents = sorted(intents, key=lambda x: float(x['probability']), reverse=True)
        # Get the tag of the intent with the highest probability
        tag = intents[0]['intent']
            #print('tag: ',tag)
        list_of_intents = intents_json['intents']
        # Find the corresponding response in intents_json
        for i in list_of_intents:
            if(i['tag'] == tag):
                #print("intent['tag']: ",i['tag'])
                result = random.choice(i['responses'])
                #print("result: ",result)
                break
        return result

    # If no intent is found, return a default response
    return "wrong response"





def chatbot_response(msg):
    print("msg: ",msg)
    ints = predict_class(msg, model)
    #print("ints: ",ints)
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
