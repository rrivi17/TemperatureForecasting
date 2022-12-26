from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from Function import PrepareData,SetArray2DCol,Evaluation,EvaluationCross
from Plotting import ResidualPlot,Graph,ScatterPlot
from sklearn.pipeline import make_pipeline


if __name__ == '__main__':
    x,y=PrepareData("JenaClimate.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)
    x_train,y_train=SetArray2DCol(x_train,y_train)
    x_test,y_test=SetArray2DCol(x_test,y_test)


    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


    model=SVR(kernel='poly',degree=3,C=0.8,epsilon=0.33,gamma=0.39)


    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)


    ResidualPlot(y_test,y_pred,file="SVR/Graph/SVRResidualPlot")
    Evaluation(y_test, y_pred, model_name='SVR',file="SVR/Evaluation/SVREvaluation")
    Graph(y_test, y_pred, file="SVR/Graph/SVRRealVSPred")
    ScatterPlot(x=y_test, y=y_pred, file="SVR/Graph/SVRScatterPlot")


    ##Evaluation CROSS
    x, y = SetArray2DCol(x, y)
    model=SVR(kernel='poly',degree=3,C=0.8,epsilon=0.33,gamma=0.39)

    model = make_pipeline(MinMaxScaler(), model)
    EvaluationCross(model,x,y,model_name='SVR',file="SVR/Evaluation/SVREvaluationCross",fileMean="SVR/Evaluation/SVREvaluationCrossMean")


