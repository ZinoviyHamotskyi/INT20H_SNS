import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA

def number_data(dataframe):
    x = dataframe[['book_pages', 'book_review_count', 'book_rating_count', 'book_rating']]
    x.book_pages = x.book_pages.apply(lambda x: str(x).split()[0])
    x.book_pages = x.book_pages.apply(lambda y: -1 if y == "nan" else int(y))
    x.book_pages[x.book_pages == -1] = x.book_pages[x.book_pages != -1].mean() 
    
    x = x.fillna(x.mean())

    feature_to_cut = ['book_pages', 'book_review_count', 'book_rating_count']

    for feature in feature_to_cut:

        x = x[(x[feature] > x[feature].quantile(0.1)) & (x[feature] < x[feature].quantile(0.999))]


    y = x.book_rating
    x = x.drop(axis=1, labels=['book_rating'])

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(x))
    data = data.rename(columns={0:'book_pages',1:'book_review_count',2:'book_rating_count'})

    decompostor = PCA(1)
    x = pd.concat([pd.DataFrame(x.book_pages).reset_index().drop(['index'], 1), pd.DataFrame(decompostor.fit_transform(data[['book_review_count', 'book_rating_count']])).reset_index().drop(['index'], 1)], 1)
    x = x.rename(columns={0:"popularity"})
    return x, y


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

