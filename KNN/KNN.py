from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
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

    model=KNeighborsRegressor(n_neighbors=11,p=1,weights='distance')
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)

    ScatterPlot(x=y_test, y=y_pred, file="KNN/Graph/KNNScatterPlot")
    Evaluation( y_test, y_pred, model_name='KNN',file="KNN/Evaluation/KNNEvaluation")
    ResidualPlot(y_test, y_pred, file="KNN/Graph/KNNResidualPlot")
    Graph(y_test, y_pred, file="KNN/Graph/KNNRealVSPred")

    x, y = SetArray2DCol(x, y)
    model=KNeighborsRegressor(n_neighbors=11,p=1,weights='distance')
    model = make_pipeline(MinMaxScaler(), model)
    EvaluationCross(model,x,y,model_name='KNN',file="KNN/Evaluation/KNNEvaluationCross",fileMean="KNN/Evaluation/KNNEvaluationCrossMean")
