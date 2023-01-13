from Function import PrepareData,SetArray2DCol,SaveGridSearch
from Plotting import PlotSplit
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit,RandomizedSearchCV



def GridSearch(x, y):
    gamma=np.random.randint(1,40,size=20)/100#def:1/13
    C=np.random.randint(1,20,size=20)/10#default 1-->ok
    epsilon=np.random.randint(1,40,size=20)/100#default 0.1
    degree=[2,3,4]#default 3


    param_grid=[{'svr__C':C,'svr__kernel':['rbf'],'svr__epsilon':epsilon,'svr__gamma':gamma},
                {'svr__C':C,'svr__kernel':['linear'],'svr__epsilon':epsilon},
                {'svr__C':C,'svr__kernel':['poly'],'svr__degree':degree,'svr__gamma':gamma,'svr__epsilon':epsilon}]

    TimeSplit=TimeSeriesSplit(n_splits=5)


    model = Pipeline(steps=[('preprocessing',MinMaxScaler()),('svr',SVR())])

    grid=RandomizedSearchCV(estimator=model,random_state=1, param_distributions=param_grid,cv=TimeSplit,n_iter=50,scoring='r2')
    grid_result = grid.fit(x, y)

    SaveGridSearch(param_grid,grid_result,folder="SVR/GridSearch")
    PlotSplit(x,y,cv=TimeSplit,file='SVR/GridSearch/TimeSerieSplit')


if __name__=='__main__':
    x, y = PrepareData("JenaClimate.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)

    x_train= SetArray2DCol(x_train)
    x_test = SetArray2DCol(x_test)

    GridSearch(x_train,y_train)
