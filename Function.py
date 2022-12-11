import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#tolgo info e warning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,r2_score,mean_absolute_error,make_scorer
from sklearn.ensemble import  RandomForestRegressor
from sktime.forecasting.model_selection import SlidingWindowSplitter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import TimeSeriesSplit,cross_validate
import sys
path=sys.path[1]
def CheckPath(file):
    if '/' not in file:
        return True
    else:#tolgo nome file
        path='/'.join(file.split('/')[:-1])
        isdir = os.path.isdir(path)
        return isdir

def SetArray2DRow(x,y):
    if len(x.shape)==3:
        n_row, time_step, n_feature = x.shape
        x = x.reshape((n_row * time_step, n_feature))  # a ogni riga associo il valore y, #(12576, 16)
        y = np.repeat(y, repeats=[time_step], axis=0)
        return(x,y)
    else:
        print("l'array inserito non è 3D")

def SetArray2DCol(x,y):
    if len(x.shape)==3:
        n_row, time_step, n_feature = x.shape
        x = x.reshape((n_row , n_feature* time_step))  # a ogni riga associo il valore y, #(12576, 16)
        #y = np.repeat(y, repeats=[time_step], axis=0)
        return(x,y)
    else:
        print("l'array inserito non è 3D")

def SaveModel(model,model_name):
    if CheckPath(f'{path}/{model_name}/SavedModel'):
        model.save(f'{path}/{model_name}/SavedModel/{model_name}.h5')  # salvo modello

def Evaluation(yu,yp,file=''):
    ev={}
    mse = mean_squared_error(yu, yp)
    ev['MSE']=mse
    rmse = np.sqrt(mse)
    ev['RMSE'] = rmse
    mae = mean_absolute_error(yu,yp)
    ev['MAE']=mae
    mape = mean_absolute_percentage_error(yu, yp)
    ev['MAPE']=mape*100
    r2 = r2_score(yu, yp)
    ev['R2']=r2


    ev = pd.DataFrame(ev.values(), index=ev.keys(), columns=['Evaluation'])
    if file != '' and CheckPath(f'{path}/{file}'):
        ev.to_csv(f'{path}/{file}.csv',float_format="%.3f")
        with open(f'{path}/{file}.txt', 'w') as f:
            ev = ev.to_string(header=True, index=True,float_format="%.3f")
            f.write(ev)
    else:
        return ev

def EvaluationCross(model,x_test,y_test,model_name,file='',fileGlobal='Evaluation/Evaluation'):
    mse = make_scorer(mean_squared_error)
    mae = make_scorer(mean_absolute_error)
    r2 = make_scorer(r2_score)
    mape = make_scorer(mean_absolute_percentage_error)
    #TimeSplit = TimeSeriesSplit(n_splits=2, test_size=int((len(y_test)) / 2 * 0.5))

    TimeSplit = TimeSeriesSplit(n_splits=5)
    ev = cross_validate(model, x_test, y_test, cv=TimeSplit, scoring={'MSE': mse, 'MAE': mae, 'MAPE': mape, 'R2': r2})

    ris = {}
    for key in ev:#moltiplica mape
        if 'test' in key:
            metric = key.split('_')[1].strip()
            ris[metric] = round(ev[key].mean(), 3)
            if metric=='MAPE':
                ris[metric] = round((ev[key]).mean(), 3)
            if metric == 'MSE':
                ris['RMSE'] = round((np.sqrt(ev[key]).mean()), 3)
    ev = pd.DataFrame(ris.values(), index=ris.keys(), columns=['Evaluation'])
    if file != '' and CheckPath(f'{path}/{file}'):
        ev.to_csv(f'{path}/{file}.csv', float_format="%.3f")
        with open(f'{path}/{file}.txt', 'w') as f:
            f.write( ev.to_string(header=True, index=True, float_format="%.3f"))
    ev = pd.DataFrame(ris.values(), index=ris.keys(), columns=['Evaluation'])
    if fileGlobal!='' and CheckPath(f'{path}/{file}'):
        EvaluationAllCross(ev,model_name,fileGlobal)
    else:
        return ev

def EvaluationAll(yu,yp,model_name,file='Evaluation/Evaluation'):
    ev = {}
    mse = mean_squared_error(yu, yp)
    ev['MSE'] = mse
    rmse = np.sqrt(mse)
    ev['RMSE'] = rmse
    mae = mean_absolute_error(yu, yp)
    ev['MAE'] = mae
    mape = mean_absolute_percentage_error(yu, yp)
    ev['MAPE'] = mape * 100
    r2 = r2_score(yu, yp)
    ev['R2'] = r2

    try:
        data = pd.read_csv(f'{path}/{file}.csv', index_col=[0])
        data[model_name]=ev.values()
    except:
        data = pd.DataFrame(ev.values(), index=ev.keys(), columns=[f'{model_name}'])

    data.to_csv(f'{path}/{file}.csv', float_format="%.3f")
    with open(f'{path}/{file}.txt', 'w') as f:
        data = data.to_string(header=True, index=True, float_format="%.3f")
        f.write(data)

def EvaluationAllCross(ev,model_name,fileGlobal='Evaluation/Evaluation'):
    try:
        data = pd.read_csv(f'{path}/{fileGlobal}.csv', index_col=[0])
        data[model_name]=ev.values
    except:
        data = pd.DataFrame(ev.values, index=ev.index, columns=[f'{model_name}'])

    data.to_csv(f'{path}/{fileGlobal}.csv', float_format="%.3f")
    with open(f'{path}/{fileGlobal}.txt', 'w') as f:
        data = data.to_string(header=True, index=True, float_format="%.3f")
        f.write(data)

def PrepareData(file,lag=3,horizon=1):
    df = pd.read_csv(f'{path}/{file}', index_col=[0])
    dates=df.index[lag:]
    dy = df["T"]
    #dx = df.drop(columns=["T (degC)", "Date Time"])
    #dx=df[['VPmax (mbar)','T (degC)','p (mbar)']]
    dx=df

    x,y = [],[]

    data_gen = TimeseriesGenerator(dx, dy, length=lag, sampling_rate=1, batch_size=horizon, stride=1)
    for lx, ly in data_gen:
        x.append(lx[0])
        y.append(ly[0])

    x = np.array(x)
    #y = np.array(y)
    dates=pd.to_datetime(dates)
    y=pd.Series(y,index=dates)
    #print(y)

    return(x,y)

def PrepareData2(file,lag=3,horizon=1):
    df = pd.read_csv(f'{path}/{file}', index_col=[0])
    dates=df.index[lag:]
    dy = df["T"]
    #dx = df.drop(columns=["T (degC)", "Date Time"])
    #dx=df[['VPmax (mbar)','T (degC)','p (mbar)']]
    dx=df

    x,y = [],[]


    splitter =  SlidingWindowSplitter(fh=horizon, window_length=lag, step_length=1)
    data_gen=splitter.split(dx,dy)
    splitter.split()
    print(list(data_gen))
    for lx, ly in data_gen:
        x.append(lx[0])
        y.append(ly[0])

    x = np.array(x)
    #y = np.array(y)
    dates=pd.to_datetime(dates)
    y=pd.Series(y,index=dates)
    print(y)

    return(x,y)

def SaveGridSearch(param_grid,grid_result,folder=''):#salvo risultati gridsearch
    single_params = []
    if type(param_grid) ==list:
        all_params = []
        for dict in param_grid:
            for key in dict:
                all_params.append(key)
        for param in set(all_params):
            if all_params.count(param) < len(param_grid):
                single_params.append(param)

    parametri = {}
    for dict in grid_result.cv_results_['params']:
        for param in single_params:
            if param not in dict:
                if param not in parametri:
                    parametri[param] = []
                parametri[param].append('-')
        for key in dict:
            if key not in parametri:
                parametri[key] = []
            try:
                parametri[key].append(round(float(dict[key]), 4))
            except:
                parametri[key].append(dict[key])

    d = {'Risultato medio': [round(float(i), 4) for i in grid_result.cv_results_['mean_test_score']], 'Std': [round(float(i), 4) for i in grid_result.cv_results_['std_test_score']]}
    d = d | parametri
    best = {'Risultato medio: ': round(float(grid_result.best_score_), 4)}

    best = best | grid_result.best_params_
    best = pd.DataFrame(best, index=["Best"])

    df = pd.DataFrame(d, index=grid_result.cv_results_['rank_test_score'])
    df = df.sort_index()


    df.to_csv(f'{path}/{folder}/Results.csv')
    with open(f'{path}/{folder}/Results.txt','w') as f:
        df = df.to_string(header=True, index=True)
        f.write(df)
    best.to_csv(f'{path}/{folder}/Best.csv')
    with open(f'{path}/{folder}/Best.txt', 'w') as f:
        best = best.to_string(header=True, index=True,float_format="%.4f")
        f.write(best)


if __name__ == '__main__':
    x, y = PrepareData("JenaClimate.csv")
    print(y[:10])

    print(len(y))
    #se cambio shape qua?