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

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemm_tokens = [lemmatizer.lemmatize(word) for word in text]
    return lemm_tokens

# I am using bag of words model to convert text data into numerical data
def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bag = [0] * len(vocab)
    for idx, word in enumerate(vocab):
        if word in tokens:
            bag[idx] = 1
    return bag

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': labels[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


    






