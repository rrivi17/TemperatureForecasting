from sklearn.model_selection import train_test_split
import numpy as np
from Function import PrepareData,SetArray2DCol,EvaluationCross,Evaluation
from Plotting import ResidualPlot,Graph,ScatterPlot
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    x,y=PrepareData("JenaClimate.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)
    model = LinearRegression()
    x_train, y_train = SetArray2DCol(x_train, y_train)
    x_test, y_test = SetArray2DCol(x_test, y_test)

    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)

    Evaluation(y_test, y_pred,model_name='LinearRegr', file="LinearRegr/Evaluation/LinearRegrEvaluation")
    ResidualPlot(y_test, y_pred, file="LinearRegr/Graph/LinearRegrResidualPlot")
    Graph(y_test, y_pred, file="LinearRegr/Graph/LinearRegrRealVSPred")
    ScatterPlot(x=y_test,y=y_pred,file="LinearRegr/Graph/LinearRegrScatterPlot")

    model = LinearRegression()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    EvaluationCross(model,x,y,model_name='LinearRegr',file="LinearRegr/Evaluation/LinearRegrEvaluationCross",fileMean="LinearRegr/Evaluation/LinearRegrEvaluationCrossMean")

