import os

# disable TensorFlow warnings not error.
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random

# for serialization
import pickle

import json
import numpy as np

# Lemmatization 词形还原 将 “running” 还原为 “run”
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
# Dense layer: normal feed forward layer, Activation layer ,Dropout layer: for regularization
from tensorflow.keras.layers import Dense, Activation, Dropout
# SGD: stochastic gradient descent minimize the loss function, implement optimization
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.load(open('intents.json'))

words = []
classes = []
documents = []
ignore_symbols = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize individual words
        word_list = nltk.word_tokenize(pattern)
        # expend() requires an iterable object as its argument. non-iterable object (like an integer directly) will raise TypeError
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_symbols]
words = sorted(set(words))
classes = sorted(classes)

# For using in Flask.
pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word) for word in word_patterns if word not in ignore_symbols]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
# training = np.array(training)
# train_x = list(training[:, 0])
# train_y = list(training[:, 1])
train_x = np.array([pair[0] for pair in training])
train_y = np.array([pair[1] for pair in training])


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save('model/chatbot_model.keras')