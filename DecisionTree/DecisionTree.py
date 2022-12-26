from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from Function import PrepareData,EvaluationCross,SetArray2DCol,Evaluation
from Plotting import Graph,ResidualPlot,ScatterPlot,PlotTree
import numpy as np



if __name__=='__main__':
    x, y = PrepareData("JenaClimate.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)

    x_train, y_train = SetArray2DCol(x_train, y_train)
    x_test, y_test = SetArray2DCol(x_test, y_test)

    model=DecisionTreeRegressor(min_samples_leaf=20,min_samples_split=9,criterion='squared_error',max_depth=20)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    Evaluation(y_test, y_pred, model_name='DecisionTree',file="DecisionTree/Evaluation/DecisionTreeEvaluation")
    ResidualPlot(y_test, y_pred, file="DecisionTree/Graph/DecisionTreeResidualPlot")
    Graph(y_test, y_pred, file="DecisionTree/Graph/DecisionTreeRealVSPred")
    ScatterPlot(x=y_test, y=y_pred, file="DecisionTree/Graph/DecisionTreeScatterPlot")
    #PlotTree(model,'DecisionTree/Graph/PlotTree')

    model = DecisionTreeRegressor(min_samples_leaf=20, min_samples_split=9, criterion='squared_error', max_depth=20)
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    EvaluationCross(model,x,y,model_name='DecisionTree',file="DecisionTree/Evaluation/DecisionTreeEvaluationCross",fileMean="DecisionTree/Evaluation/DecisionTreeEvaluationCrossMean")



    #PrintTree(model)
    #tree.plot_tree(model)
    #fa plot albero
    #fig=plt.figure()
    #tree.plot_tree(model, filled=True)#, feature_names=x.columns)
    #fig.savefig("PlotTree.pdf")