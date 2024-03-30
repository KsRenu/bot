
import nltk

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

"""# **Preprocessing**"""

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read() # read json file
intents = json.loads(data_file) # load json file


for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)# add each elements into list
        #combination between patterns and intents
        documents.append((w, intent['tag']))#add single element into end of list
        # add to tag in our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


#nltk.download('omw-1.4')

# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents\n", documents, "\n")
# classes = intents[tag]
print (len(classes), "classes\n", classes, "\n")
# words = all words, vocabulary
print (len(words), "unique lemmatized words\n", words, "\n")
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

"""# **Training Model**"""


training = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words
    pattern_words = doc[0]
    # convert pattern_words in lower case
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create bag of words array, if word match found in current pattern then put 1 otherwise 0.
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # in output array 0 value for each tag and 1 value for matched tag.
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Convert training to a NumPy array
training = np.array(training, dtype=object)

# shuffle our features and turn into np.array
random.shuffle(training)
training = list(training)

# create train and test lists. X - patterns, Y - intents
train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

from tensorflow.python.framework import ops
ops.reset_default_graph()

"""# **Build the model**"""

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
print("First layer:",model.layers[0].get_weights()[0])

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)

print("model created")


from flask import Flask, render_template, request

def clean_up_sentence(sentence):

        # tokenize the pattern - split words into array

        sentence_words = nltk.word_tokenize(sentence)
        #print(sentence_words)
        # stem each word - create short form for word

        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        #print(sentence_words)

        return sentence_words

def bow(sentence, words, show_details=True):

        # tokenize the pattern

        sentence_words = clean_up_sentence(sentence)
        #print(sentence_words)

        # bag of words - matrix of N words, vocabulary matrix

        bag = [0]*len(words)
        #print(bag)

        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
                    #print ("found in bag: %s" % w)
        #print(bag)
        return(np.array(bag))


def predict_class(sentence, model):

        # filter out predictions below a threshold

        p = bow(sentence, words,show_details=False)
        #print(p)

        res = model.predict(np.array([p]))[0]
        #print(res)

        ERROR_THRESHOLD = 0.25

        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        #print(results)
        # sort by strength of probability

        results.sort(key=lambda x: x[1], reverse=True)
        #print(results)

        return_list = []

        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

        return return_list
        #print(return_list)


def getResponse(ints, intents_json):

        tag = ints[0]['intent']
        #print(tag)

        list_of_intents = intents_json['intents']
        #print(list_of_intents)

        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result

def chatbot_response(text):
        ints = predict_class(text, model)

        #print(ints)

        res = getResponse(ints, intents)
        #print(res)
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
