import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import warnings
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

app = Flask(__name__)

# Configure file upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCmkCUuORIyV00QkWWMdzUbR3DQqv358bM"

# Load NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load and compile the model
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def home():
    return render_template('new.html')

@app.route('/get')
def get_bot_response():
    user_msg = request.args.get('msg')
    response = chatbot_response(user_msg)
    return str(response)

@app.route('/process_data', methods=['POST'])
def process_data():
    # Receive file from the user
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(filepath)
        pages = loader.load_and_split()

        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create FAISS index from PDF documents
        db = FAISS.from_documents(pages, embeddings)

        # Receive input query from the user
        query = request.form['query']

        # Query for text
        docs = db.similarity_search(query)

        # Get page content from similar documents
        content = "\n".join([x.page_content for x in docs])

        # Prepare input text for generative AI
        qa_prompt = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.----------------"
        input_text = f"{qa_prompt}\nContext:{content}\nUser question:\n{query}"

        # Initialize Google Generative AI model
        llm = ChatGoogleGenerativeAI(model="gemini-pro")

        # Generate response using Google Generative AI
        result = llm.invoke(input_text)

        return jsonify({'result': result.content})
    else:
        return jsonify({'error': 'Invalid file format'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

if __name__ == '__main__':
    app.run(port=8000)
