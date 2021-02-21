import pickle
import pandas as pd
import numpy as np
import re
import nltk
from scipy import sparse
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords') 

from settings.constants import FORMAT_VECTORIZER, GENRE_VECTORIZER, TEXT_VECTORIZER, AUTOR_VECTORIZER

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stopword_set]
    
    #stemmed words
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in meaningful_words]
    
    #join the cleaned words in a list
    cleaned_word_list = " ".join(stemmed_words)

    return cleaned_word_list

class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        self.dataset['book_authors'] = self.dataset['book_authors'].apply(lambda x: str(x).replace(' ', "_").replace('|', " ").lower())
        self.dataset.book_genre = self.dataset.book_genre.apply(lambda x: x.replace(' ', "_").replace('|', " ").lower() if str(x) != "nan" else '')
        self.dataset.book_desc = self.dataset.book_desc.apply(lambda line: preprocess(line))
        self.dataset.book_title = self.dataset.book_title.apply(lambda line: preprocess(line))
        
        
        text_vectorizer = pickle.load(open(TEXT_VECTORIZER, 'rb'))
        genre_vectorizer = pickle.load(open(GENRE_VECTORIZER, 'rb'))
        format_vectorizer = pickle.load(open(FORMAT_VECTORIZER, 'rb'))
        autor_vectorizer = pickle.load(open(AUTOR_VECTORIZER, 'rb'))

        test_text_data = self.dataset.book_desc
        X_text_test = text_vectorizer.transform(test_text_data)
        test_genre_data = self.dataset.book_genre
        X_genre_test = genre_vectorizer.transform(test_genre_data)
        test_autor_data = self.dataset.book_authors
        X_autor_test = autor_vectorizer.transform(test_autor_data)
        test_format_data = self.dataset.book_format
        X_format_test = format_vectorizer.transform(test_format_data)
        X_test = sparse.hstack([X_text_test, X_genre_test, X_autor_test, X_format_test])
        self.dataset = X_test
        return self.dataset