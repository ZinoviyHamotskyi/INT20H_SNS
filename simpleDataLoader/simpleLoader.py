import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns

df = pd.read_csv('train.csv')

print(df.info())
print(missing_values_table(df))

df = df.dropna()
print(df.nunique())

y = df['book_rating']
x = df[['book_format', 'book_pages', 'book_review_count', 'book_rating_count']]

#pages to int
x.loc['book_pages'] = x.book_pages.apply(lambda x: int(x.split()[0]))

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

print(x.head())
print(y.head())

simple = pd.concat([x, y], axis=1)
print(simple.head())
simple.to_csv('simple.csv', sep=',')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print('shape of x_train: ', x_train.shape)
print('shape of x_test: ', x_test.shape)
print('shape of y_train: ', y_train.shape)
print('shape of y_test: ', y_test.shape)


