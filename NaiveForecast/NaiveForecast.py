import pandas as pd
from Function import PrepareData,Evaluation,EvaluationAll,EvaluationAllSingle
from sklearn.model_selection import train_test_split
from Plotting import ResidualPlot,Graph,ScatterPlot
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,r2_score,mean_absolute_error
import os
import sys
path=sys.path[1]
from sklearn.model_selection import TimeSeriesSplit

def CheckPath(file):
    if '/' not in file:
        return True
    else:
        path='/'.join(file.split('/')[:-1])
        isdir = os.path.isdir(path)
        return isdir


def CrossValNaive(x,cv,y_pred,model_name='NaiveForecast',fileMean='',fileGlobal='EvaluationCross/EvaluationCrossMean',file=''):

    ev={'MSE':[],'RMSE':[],'MAE':[],'MAPE':[],'R2':[]}
    for train, test in cv.split(x):  # calcolo tutte le previsioni
        ev['MSE'].append(mean_squared_error(y[test[0]:test[-1]],y_pred[test[0]:test[-1]]))
        ev['RMSE'].append(np.sqrt(mean_squared_error(y[test[0]:test[-1]],y_pred[test[0]:test[-1]])))
        ev['MAE'].append(mean_absolute_error(y[test[0]:test[-1]],y_pred[test[0]:test[-1]]))
        ev['MAPE'].append(mean_absolute_percentage_error(y[test[0]:test[-1]],y_pred[test[0]:test[-1]]))
        ev['R2'].append(r2_score(y[test[0]:test[-1]],y_pred[test[0]:test[-1]]))

    #salvo valori singoli
    ris = pd.DataFrame(ev.values(), index=ev.keys(), columns=['CV' + str(i) for i in range(1, len(ev['MSE']) + 1)])
    EvaluationAllSingle(ris,model_name='NaiveForecast')#salva nel globale

    if file != '' and CheckPath(f'{path}/{file}'):
        ris.to_csv(f'{path}/{file}.csv', float_format="%.3f")
        with open(f'{path}/{file}.txt', 'w') as f:
            f.write(ris.to_string(header=True, index=True, float_format="%.3f"))
    else:
        return ev


    ris = {}
    for metric in ev:
        ris[metric] = sum(ev[metric]) / len(ev[metric])#calcolo valori medi
    ev = pd.DataFrame(ris.values(), index=ris.keys(), columns=['Evaluation'])

    if file != '' and CheckPath(f'{path}/{fileMean}'):
        ev.to_csv(f'{path}/{fileMean}.csv', float_format="%.3f")
        with open(f'{path}/{fileMean}.txt', 'w') as f:
            f.write(ev.to_string(header=True, index=True, float_format="%.3f"))

    if fileGlobal != '' and CheckPath(f'{path}/{file}'):
        EvaluationAll(ev, model_name, fileGlobal)

    else:
        return ev


if __name__ == '__main__':
    x,y=PrepareData("JenaClimate.csv")

    y_pred = pd.Series(y).shift(1)

    y = y[1:]
    y_pred = y_pred[1:]

    y_pred_train, y_pred_test = train_test_split(y_pred, test_size=0.3,random_state=2, shuffle=False)
    y_train, y_test = train_test_split(y, test_size=0.3, shuffle=False)

    Evaluation(y_test, y_pred_test,model_name='NaiveForecast', file="NaiveForecast/Evaluation/NaiveForecastEvaluation")
    ResidualPlot(y_test, y_pred_test, file="NaiveForecast/Graph/NaiveForecastResidualPlot")
    Graph(y_test,y_pred_test,file="NaiveForecast/Graph/NaiveForecastRealVSPred")
    ScatterPlot(x=y_test, y=y_pred_test, file="NaiveForecast/Graph/NaiveForecastScatterPlot")


    CrossValNaive(x,cv=TimeSeriesSplit(n_splits=5),y_pred=y_pred,fileMean="NaiveForecast/Evaluation/NaiveForecastEvaluationCrossMean",file="NaiveForecast/Evaluation/NaiveForecastEvaluationCross")
