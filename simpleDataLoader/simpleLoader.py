import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#def missing_values_table(df):
#    mis_val = df.isnull().sum()
#    mis_val_percent = 100 * df.isnull().sum() / len(df)
#    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
#
#    mis_val_table_ren_columns = mis_val_table.rename(
#        columns={0: 'Missing Values', 1: '% of Total Values'})
#    mis_val_table_ren_columns = mis_val_table_ren_columns[
#        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
#        '% of Total Values', ascending=False).round(1)

#    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
#                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
#          " columns that have missing values.")
#    return mis_val_table_ren_columns

#df = pd.read_csv('train.csv')

#print(df.info())
#print(missing_values_table(df))

#df = df.dropna()
#print(df.nunique())

#y = df['book_rating']
#x = df[['book_format', 'book_pages', 'book_review_count', 'book_rating_count']]

#pages to int
#x['book_pages'] = x.book_pages.apply(lambda x: int(x.split()[0]))

#one-hot encoding
#encoder = OneHotEncoder()
#matrix = pd.DataFrame.sparse.from_spmatrix(encoder.fit_transform(pd.DataFrame(x.book_format)))
#feature_names = list(encoder.get_feature_names())
#columns = {i: feature_names[i] for i in range(107)}
#matrix = matrix.rename(columns=columns)
#x = x.reset_index().drop("index", 1)
#concating
#x = pd.concat([x, matrix], 1)
#del x["book_format"]

#print(x.head())
#print(y.head())

#simple = pd.concat([x, y], axis=1)
#print(simple.head())
#simple.to_csv('simple.csv', sep=',')

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

#x_train = pd.concat([x_train.reset_index().drop("index", 1), y_train.reset_index().drop("index", 1)], 1)
#x_selected = x_train[(x_train.book_pages > x_train.book_pages.quantile(0.05)) | (x_train.book_pages < x_train.book_pages.quantile(0.95))]

#x_train = x_selected.drop(axis=1, labels=['book_rating'])
#y_train = x_selected.book_rating

#print('shape of x_train: ', x_train.shape)
#print('shape of x_test: ', x_test.shape)
#print('shape of y_train: ', y_train.shape)
#print('shape of y_test: ', y_test.shape)


def cleandata(dataframe):

    dataframe = dataframe.dropna()

    y = dataframe['book_rating']
    x = dataframe[['book_format', 'book_pages', 'book_review_count', 'book_rating_count']]
    x['book_pages'] = x.book_pages.apply(lambda x: int(x.split()[0]))

    #one-hot encoding
    encoder = OneHotEncoder()
    matrix = pd.DataFrame.sparse.from_spmatrix(encoder.fit_transform(pd.DataFrame(x.book_format)))
    feature_names = list(encoder.get_feature_names())
    columns = {i: feature_names[i] for i in range(107)}
    matrix = matrix.rename(columns=columns)
    x = x.reset_index().drop("index", 1)

    #concating
    x = pd.concat([x, matrix], 1)
    del x["book_format"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

    x_train = pd.concat([x_train.reset_index().drop("index", 1), y_train.reset_index().drop("index", 1)], 1)
    feature_to_cut = ['book_pages', 'book_review_count', 'book_rating_count']

    for feature in feature_to_cut:

        x_train = x_train[(x_train[feature] > x_train[feature].quantile(0.1)) & (x_train[feature] < x_train[feature].quantile(0.999))]


    y_train = x_train.book_rating
    x_train = x_train.drop(axis=1, labels=['book_rating'])

    return x_train, x_test, y_train, y_test
