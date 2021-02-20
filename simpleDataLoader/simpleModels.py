import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, BayesianRidge
import simpleLoader
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('train.csv')
df.info()
temp = pd.concat([df['book_review_count'], df['book_rating_count']], axis=1)
print(temp.corr())
'''
simple = pd.concat([x, y], axis=1)
print(simple.head())
simple.to_csv('simple.csv', sep=',')
'''
x, y = simpleLoader.book_format_data(df)
print(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)


print('shape of x_train: ', x_train.shape)
print('shape of x_test: ', x_test.shape)
print('shape of y_train: ', y_train.shape)
print('shape of y_test: ', y_test.shape)



def bigspace(name):
    print('####################################################')
    print('##################', name, '#####################')
    print('####################################################')
def space():
    print('------------------------------------------------------')
result = []
models1 = [LinearRegression(),
           SGDRegressor(),
           RidgeCV(alphas=[0.1, 0.25, 0.5, 0.75, 1]),
           RidgeCV(alphas=[1, 2.5, 5, 7.5, 10]),
           RidgeCV(alphas=[100, 250, 500, 750, 1000]),
           BayesianRidge()]

bigspace('  Linear Regression Models  ')
'''

'''
for model in models1:
    model.fit(x_train.to_numpy(), y_train)
    result.append({'name': model.__str__(),
                   'score': mean_squared_error(y_test, model.predict(x_test), squared=False)})

result = pd.DataFrame(result)
print(result)
result = []

bigspace(' Ensemble of Models  ')
cleanEnsembles = [GradientBoostingRegressor(n_estimators=200),
                  AdaBoostRegressor(base_estimator=LinearRegression(), n_estimators=200),
                  AdaBoostRegressor(n_estimators=200),
                  BaggingRegressor(n_estimators=200),
                  RandomForestRegressor(n_estimators=200),
                  ExtraTreesRegressor(n_estimators=200)]
'''
'''
for model in cleanEnsembles:
    model.fit(x_train, y_train)
    result.append({'name': model.__str__(),
                   'score': mean_squared_error(y_test, model.predict(x_test), squared=False)})

result = pd.DataFrame(result)
print(result)
result = []

neural_nets = [MLPRegressor(hidden_layer_sizes=50),
               MLPRegressor(hidden_layer_sizes=100),
               MLPRegressor(hidden_layer_sizes=20),
               MLPRegressor(hidden_layer_sizes=5),
               MLPRegressor(max_iter=100, alpha=1e-4),
               MLPRegressor(max_iter=300, alpha=1e-5),
               MLPRegressor(max_iter=500, alpha=1e-7),
               MLPRegressor(solver='lbfgs'),
               MLPRegressor(solver='sgd'),
               MLPRegressor()]

bigspace(' Neural Networks  ')


for model in neural_nets:
    model.fit(x_train, y_train)
    result.append({'name': model.__str__(),
                   'score': mean_squared_error(y_test, model.predict(x_test), squared=False)})


result = pd.DataFrame(result)
print(result)
