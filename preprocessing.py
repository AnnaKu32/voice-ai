from nltk.tokenize import NLTKWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

import re
import nltk
nltk.download('stopwords')

SYMBOLS = '{}()[]\\.,:;+-_@*/#&$...…|<>=~^!?”“’"'

class Preprocessor:
    def __init__(self):
        self.tokenizer = NLTKWordTokenizer()
        self.lm = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def expand_contractions(self, text):
        contractions = {
            "isn't": 'is not',
            "he's": 'he is',
            "wasn't": 'was not',
            "there's": 'there is',
            "couldn't": 'could not',
            "won't": 'will not',
            "they're": 'they are',
            "she's": 'she is',
            "wouldn't": 'would not',
            "haven't": 'have not',
            "you've": 'you have',
            "what's": 'what is',
            "weren't": 'were not',
            "we're": 'we are',
            "hasn't": 'has not',
            "you'd": 'you would',
            "shouldn't": 'should not',
            "let's": 'let us',
            "they've": 'they have',
            "i'm": 'i am',
            "im": 'i am',
            "we've": 'we have',
            "it's": 'it is',
            "don't": 'do not',
            "that's": 'that is',
            "i'm": 'i am',
            "it's": 'it is',
            "she's": 'she is',
            "he's": 'he is',
            "i'm": 'i am',
            "i'd": 'i would',
            "he's": 'he is',
            "there's": 'there is'
        }

        text = text.lower()
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text)
        return text

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
    
    def lem_words(self, text):
        return [self.lm.lemmatize(word) for word in text]


    # def remove_stopwords(self, text):
    #     return [w for w in text if not w.lower() in self.stop_words]

   
    def preprocess(self, text):
        text = self.expand_contractions(text)
        text = self.tokenize(text)
        text = self.remove_stopwords(text)
        return text