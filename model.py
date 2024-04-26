# Description: This file is used to train the bag-of-words(bow) model using the intents.json file

import tensorflow
import keras
from keras import layers, optimizers

import nltk
from nltk.stem import WordNetLemmatizer, LancasterStemmer
nltk.download('punkt')
nltk.download('wordnet')

import json 
import numpy as np
import random


intents = json.loads(open('intents.json').read())

# ---------------------------------- Preprocessing ----------------------------------
words = []
classes = []
data_x = []
data_y = []
ignore_words = ['?', '!', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_x.append(tokens)
        data_y.append(intent['tag'])
        
    if intent['tag'] not in classes:
        classes.append(intent['tag'])


lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))


# ---------------------------------- Training Data ----------------------------------
training = []
output_empty = [0] * len(classes)

for idx, doc in enumerate(data_x):
    bag = []
    text = [lemmatizer.lemmatize(w.lower()) for w in doc]
    text = [stemmer.stem(w.lower()) for w in doc]
    
    for w in words:
        bag.append(1) if w in text else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(data_y[idx])] = 1
    
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])


# ---------------------------------- Model ----------------------------------
model = keras.Sequential()
model.add(layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(train_y[0]), activation='softmax'))

adam = optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', 
              optimizer=adam, 
              metrics=['accuracy'])
print(model.summary())
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)


