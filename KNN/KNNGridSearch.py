from sklearn.model_selection import train_test_split
from Function import PrepareData,SetArray2DCol,SaveGridSearch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline


def GridSearch(x, y):
    n_neighbors = list(range(1, 21))

    weights=['uniform', 'distance']
    p=[1,2]


    param_grid=[{'KNN__n_neighbors':n_neighbors, 'KNN__weights':weights,'KNN__p':p,'KNN__metric':['minkowski']},
                {'KNN__n_neighbors': n_neighbors,'KNN__weights': weights, 'KNN__metric': ['chebyshev']} ]

    TimeSplit = TimeSeriesSplit(n_splits=5)

    model = Pipeline(steps=[('preprocessing',MinMaxScaler()),('KNN',KNeighborsRegressor())])
    grid = GridSearchCV(estimator= model, verbose=1, param_grid=param_grid, cv=TimeSplit, scoring='r2')

    grid_result = grid.fit(x, y)

    SaveGridSearch(param_grid, grid_result, folder="KNN/GridSearch")

if __name__ == '__main__':
    x,y=PrepareData("JenaClimate.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)
    x_train=SetArray2DCol(x_train)
    x_test=SetArray2DCol(x_test)

    GridSearch(x_train, y_train)