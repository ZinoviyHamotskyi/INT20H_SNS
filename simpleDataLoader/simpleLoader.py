import pandas as pd
from sklearn.model_selection import train_test_split

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
print(df.info())

y = df['book_rating']
x = df[['book_format', 'book_pages', 'book_review_count', 'book_rating_count']]

print(x.head())
print(y.head())


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print('shape of x_train: ', x_train.shape)
print('shape of x_test: ', x_test.shape)
print('shape of y_train: ', y_train.shape)
print('shape of y_test: ', y_test.shape)