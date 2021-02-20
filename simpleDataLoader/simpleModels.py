import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, BayesianRidge
import simpleLoader



pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('train.csv')
'''
simple = pd.concat([x, y], axis=1)
print(simple.head())
simple.to_csv('simple.csv', sep=',')
'''
x_train, x_test, y_train, y_test = simpleLoader.cleandata(df)


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
    space()
    model.fit(x_train, y_train)
    print(model)
    print(mean_squared_error(y_test, model.predict(x_test), squared=False))
    result.append({'name': model.__str__(),
                   'score': mean_squared_error(y_test, model.predict(x_test), squared=False)})
    space()

bigspace(' Ensemble of Models  ')
cleanEnsembles = [GradientBoostingRegressor(n_estimators=200),
                  AdaBoostRegressor(base_estimator=LinearRegression(), n_estimators=200),
                  AdaBoostRegressor(n_estimators=200),
                  BaggingRegressor(n_estimators=200),
                  RandomForestRegressor(n_estimators=200),
                  ExtraTreesRegressor(n_estimators=200)]
result = pd.DataFrame(result)
print(result)
for model in cleanEnsembles:
    space()
    model.fit(x_train, y_train)
    print(model)
    print(mean_squared_error(y_test, model.predict(x_test), squared=False))
    space()
    result.append({'name': model.__str__(),
                   'score': mean_squared_error(y_test, model.predict(x_test), squared=False)})

