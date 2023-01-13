import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from Function import PrepareData,EvaluationCross,SetArray2DCol,Evaluation
from Plotting import Graph,ResidualPlot,ScatterPlot


if __name__ == '__main__':
    x, y = PrepareData("JenaClimate.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)

    x_train = SetArray2DCol(x_train)
    x_test = SetArray2DCol(x_test)

    model = RandomForestRegressor(criterion="squared_error", random_state=1, min_samples_split=3,min_samples_leaf=9, n_estimators=300)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)


    Evaluation(y_test, y_pred, model_name='RandomForest', file="RandomForest/Evaluation/RandomForestEvaluation",)
    ResidualPlot(y_test,y_pred,file="RandomForest/Graph/RandomForestResidualPlot")
    Graph(y_test, y_pred, file="RandomForest/Graph/RandomForestRealVSPred")
    ScatterPlot(x=y_test, y=y_pred, file="RandomForest/Graph/RandomForestScatterPlot")


    model = RandomForestRegressor(criterion="squared_error", random_state=1, min_samples_split=3, min_samples_leaf=9,n_estimators=300)
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    EvaluationCross(model,x,y,model_name='RandomForest',file="RandomForest/Evaluation/RandomForestEvaluationCross",fileMean="RandomForest/Evaluation/RandomForestEvaluationCrossMean")

