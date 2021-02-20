import numpy as np
import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LinearRegression
import scipy.sparse as sps
import scipy.sparse
from random import sample
import math
import statsmodels.api as sm

def preprocess(raw_text):
    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stopword_set]

    # stemmed words
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in meaningful_words]

    # join the cleaned words in a list
    cleaned_word_list = " ".join(stemmed_words)

    return cleaned_word_list

def preprocessFrame(dataframe):
    y = raw_data['book_rating']
    dataframe = dataframe.apply(lambda line: preprocess(line))

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.7, ngram_range=(1, 5))
    vectorizer.fit(dataframe)

    X = vectorizer.transform(dataframe)
    X = pd.DataFrame.sparse.from_spmatrix(X)
    return X,y

def linearreg(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    skm = LinearRegression().fit(x_train, y_train.to_frame())
    scores = skm.score(x_train, y_train.to_frame())
    print(skm.intercept_, skm.coef_)
    print(scores)

raw_data = pd.read_csv('train.csv')
print("----description----")
desc_data = raw_data['book_desc']
X, y = preprocessFrame(desc_data)
linearreg(X, y)




