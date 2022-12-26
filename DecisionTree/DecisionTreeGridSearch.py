from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from Function import PrepareData,SetArray2DCol,SaveGridSearch
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit, RandomizedSearchCV

def GridSearch(x, y):
    criterion = ['squared_error', 'absolute_error']
    max_depth=[None,10,20,30,50]
    min_samples_split = list(range(2, 21))# #The minimum number of samples required to split an internal node,default = 2
    min_samples_leaf=list(range(1, 21))#21
    # #The minimum number of samples required to be at a leaf node.
    # A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches

    param_grid={'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,'max_depth':max_depth,
                'criterion':criterion}

    TimeSplit = TimeSeriesSplit(n_splits=5)

    grid = RandomizedSearchCV(estimator=DecisionTreeRegressor(), verbose=1, param_distributions=param_grid, cv=TimeSplit, n_iter=150,scoring='r2')

    #grid = GridSearchCV(estimator=DecisionTreeRegressor(random_state=1), verbose=1, param_grid=param_grid, cv=TimeSplit,scoring='r2')
    grid_result = grid.fit(x, y)

    SaveGridSearch(param_grid, grid_result, folder="DecisionTree/GridSearch")


if __name__=='__main__':
    x, y = PrepareData("JenaClimate.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)

    x_train, y_train = SetArray2DCol(x_train, y_train)
    x_test, y_test = SetArray2DCol(x_test, y_test)

    GridSearch(x_train,y_train)
