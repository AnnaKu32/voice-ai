from model import model

import numpy as np
import re
import random
import nltk
from nltk.tokenize import NLTKWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('stopwords')

SYMBOLS = '{}()[]\\.,:;+-_@*/#&$...…|<>=~^!?”“’"'
STOPWORDS = set(stopwords.words('english'))
ERROR_THRESHOLD = 0.25

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemm_tokens = [lemmatizer.lemmatize(word) for word in text]
    return lemm_tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bag = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bag[idx] = 1
    return np.array(bag)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "I'm sorry, I don't understand."
    else:
        tag = intents_list[0]
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result


    






