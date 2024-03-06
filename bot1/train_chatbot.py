import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import json
import pickle
import numpy as np

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load data from the intents file
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Lists to store words, classes, and training data
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((" ".join(w), intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)

# Process documents
for doc in documents:
    training.append((doc[0], doc[1]))

# Shuffle the training data
random.shuffle(training)

# Create Tokenizer and fit on the documents
tokenizer = Tokenizer()
tokenizer.fit_on_texts([pair[0] for pair in training])

# Convert text data to sequences and pad them
sequences = tokenizer.texts_to_sequences([pair[0] for pair in training])
padded_sequences = pad_sequences(sequences)

# Create one-hot encoding for classes
classes_one_hot = np.zeros((len(training), len(classes)))
for i, (_, intent) in enumerate(training):
    classes_one_hot[i, classes.index(intent)] = 1

# Combine padded sequences and one-hot encoded classes into the training array
training_array = [(padded_sequences[i], classes_one_hot[i]) for i in range(len(training))]

# Convert to NumPy array
training = np.array(training_array, dtype=object)

# Print information about the training data
print("Training data created")
print(f"Number of unique lemmatized words: {len(words)}")
print(f"Number of classes: {len(classes)}")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(padded_sequences.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Use legacy SGD optimizer without decay
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
hist = model.fit(padded_sequences, classes_one_hot, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5', hist)
print("Model created")

