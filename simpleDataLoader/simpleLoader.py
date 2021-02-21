import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def number_data(dataframe):
    x = dataframe[['book_pages', 'book_review_count', 'book_rating_count', 'book_rating']]
    x = x.dropna()
    x.book_pages = x.book_pages.apply(lambda x: int(str(x).split()[0]))
    
    feature_to_cut = ['book_pages', 'book_review_count', 'book_rating_count']
    lower = {}
    upper = {}
    for feature in feature_to_cut:
        lower[feature] = x[feature].quantile(0.01)
        upper[feature] = x[feature].quantile(0.999)
        x = x[(x[feature] > x[feature].quantile(0.01)) & (x[feature] < x[feature].quantile(0.999))]


    y = x.book_rating
    x = x.drop(axis=1, labels=['book_rating'])

    return x, y, lower, upper


def number_data_test(dataframe, lower, upper):
    x = dataframe[['book_pages', 'book_review_count', 'book_rating_count']]
    x = x.fillna(0)
    x.book_pages = x.book_pages.apply(lambda x: int(str(x).split()[0]))
    

    feature_to_cut = ['book_pages', 'book_review_count', 'book_rating_count']

    for feature in feature_to_cut:
        x[x[feature] < lower[feature]][feature] = lower[feature]
        x[x[feature] > upper[feature]][feature] = upper[feature]

    return x

def book_format_data(dataframe):
    x = dataframe['book_format']
    y = dataframe['book_rating']
    x = x.fillna('')
    #one-hot encoding
    encoder = OneHotEncoder()
    matrix = pd.DataFrame.sparse.from_spmatrix(encoder.fit_transform(pd.DataFrame(x)))
    feature_names = list(encoder.get_feature_names())
    columns = {i: feature_names[i] for i in range(len(feature_names))}
    matrix = matrix.rename(columns=columns)

    return matrix, y

