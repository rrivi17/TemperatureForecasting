import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#tolgo info e warning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,r2_score,mean_absolute_error,make_scorer
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import TimeSeriesSplit,cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
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
        return "l'array inserito non è 3D"

def SetArray2DCol(x,y):
    if len(x.shape)==3:
        n_row, time_step, n_feature = x.shape
        x = x.reshape((n_row , n_feature* time_step))  # a ogni riga associo il valore y, #(12576, 16)
        #y = np.repeat(y, repeats=[time_step], axis=0)
        return(x,y)
    else:
        return "l'array inserito non è 3D"

def SaveModel(model,model_name):
    if CheckPath(f'{path}/{model_name}/SavedModel'):
        model.save(f'{path}/{model_name}/SavedModel/{model_name}.h5')  # salvo modello

def Evaluation(yu,yp,model_name,file='',fileGlobal='Evaluation/Evaluation'):
    ev={}
    mse = mean_squared_error(yu, yp)
    ev['MSE']=mse
    rmse = np.sqrt(mse)
    ev['RMSE'] = rmse
    mae = mean_absolute_error(yu,yp)
    ev['MAE']=mae
    mape = mean_absolute_percentage_error(yu, yp)
    ev['MAPE']=mape#*100
    r2 = r2_score(yu, yp)
    ev['R2']=r2


    ev = pd.DataFrame(ev.values(), index=ev.keys(), columns=['Evaluation'])
    if file != '' and CheckPath(f'{path}/{file}'):
        ev.to_csv(f'{path}/{file}.csv',float_format="%.3f")
        with open(f'{path}/{file}.txt', 'w') as f:
            f.write(ev.to_string(header=True, index=True,float_format="%.3f"))
    else:
        return ev
    #ev = pd.DataFrame(ev.values(), index=ris.keys(), columns=['Evaluation'])
    if fileGlobal != '' and CheckPath(f'{path}/{file}'):
        EvaluationAll(ev, model_name, fileGlobal)

def PredCross(x,models,cv):
    y_pred = []
    i = 0
    for train, test in cv.split(x):  # calcolo tutte le previsioni
        arr = models[i].predict(x[test[0]:test[-1]])
        for val in arr:
            y_pred.append(val)
    return y_pred

def EvaluationCross(model,x,y,model_name,file='',fileGlobal='EvaluationCross/EvaluationCross',n_split=5,mean=True):
    '''mean =true se aggrego valori con la media false se calcolo tutte le previsioni e le aggrego prima di calcolare metriche'''
    #y dataframe con date
    mse = make_scorer(mean_squared_error)
    mae = make_scorer(mean_absolute_error)
    r2 = make_scorer(r2_score)
    mape = make_scorer(mean_absolute_percentage_error)

    TimeSplit = TimeSeriesSplit(n_splits=n_split)
    #PlotSplit(x,TimeSplit)

    ev = cross_validate(model, x, y, cv=TimeSplit, scoring={'MSE': mse, 'MAE': mae, 'MAPE': mape, 'R2': r2},return_estimator=True)

    if mean==True:#alcolo valori medi
        ris = {}
        for key in ev:
            if 'test' in key:
                metric = key.split('_')[1].strip()
                ris[metric] = round(ev[key].mean(), 3)
                if metric == 'MAPE':
                    ris[metric] = round((ev[key]).mean(), 3)
                if metric == 'MSE':
                    ris['RMSE'] = round((np.sqrt(ev[key]).mean()), 3)
        ev = pd.DataFrame(ris.values(), index=ris.keys(), columns=['Evaluation'])
        #if file != '' and CheckPath(f'{path}/{fileGlobal}'):
        EvaluationAll(ev, model_name, fileGlobal)

    elif mean==False:#salvo risultati tutto fold, prova
         '''y_pred=PredCross(x,models=ev['estimator'],cv=TimeSplit)
         y=y[-len(y_pred):]
         ris = Evaluation(y, y_pred, model_name='prova')'''
         ris={}
         for key in ev:
             if 'test' in key:
                 metric = key.split('_')[1].strip()
                 ris[metric]=ev[key]
                 if metric == 'MSE':
                     ris['RMSE'] = np.round(np.sqrt(ev[key]),3)
         ev = pd.DataFrame(ris.values(), index=ris.keys(), columns=['CV'+str(i) for i in range(1,len(ris['MSE'])+1)])
         EvaluationAllSingle(ev, model_name)

    if file != '' and CheckPath(f'{path}/{file}'):
        ev.to_csv(f'{path}/{file}.csv', float_format="%.3f")
        with open(f'{path}/{file}.txt', 'w') as f:
            f.write(ev.to_string(header=True, index=True, float_format="%.3f"))
    else:
        return ev


def EvaluationAllSingle(ev,model_name,fileGlobal='EvaluationCross/EvaluationCrossSingle'):
    for metric in ev.index:
        ris=ev[ev.index==metric]
        ris=pd.DataFrame(ris.values,columns=ev.columns,index=[f'{model_name}'])

        try:
            data = pd.read_csv(f'{path}/{fileGlobal}/{metric}.csv', index_col=[0])
            if model_name in data.index:
                data.loc[model_name]=ris.values
            else:
                data=pd.concat([data,ris])
        except:
            data = pd.DataFrame(ris.values, index=ris.index, columns=ris.columns)

        if fileGlobal != '' and CheckPath(f'{path}/{fileGlobal}'):
            data.to_csv(f'{path}/{fileGlobal}/{metric}.csv', float_format="%.3f")
            with open(f'{path}/{fileGlobal}/{metric}.txt', 'w') as f:
                data = data.to_string(header=True, index=True, float_format="%.3f")
                f.write(data)
        else:
            return data

def EvaluationAll(ev,model_name,fileGlobal='Evaluation/Evaluation'):
    try:
        data = pd.read_csv(f'{path}/{fileGlobal}.csv', index_col=[0])
        data[model_name]=ev.values
    except:
        data = pd.DataFrame(ev.values, index=ev.index, columns=[f'{model_name}'])

    if fileGlobal != '' and CheckPath(f'{path}/{fileGlobal}'):
        data.to_csv(f'{path}/{fileGlobal}.csv', float_format="%.3f")
        with open(f'{path}/{fileGlobal}.txt', 'w') as f:
            data = data.to_string(header=True, index=True, float_format="%.3f")
            f.write(data)
    else:
        return data

def PrepareData(file,lag=3,horizon=1,target='T'):
    df = pd.read_csv(f'{path}/{file}', index_col=[0])
    dates=df.index[lag:]
    dy = df[target]
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
                parametri[key].append(round(float(dict[key]), 3))
            except:
                parametri[key].append(dict[key])

    d = {'Risultato medio': [round(float(i), 4) for i in grid_result.cv_results_['mean_test_score']], 'Std': [round(float(i), 4) for i in grid_result.cv_results_['std_test_score']]}
    d = d | parametri

    best = {'Risultato medio: ': round(float(grid_result.best_score_), 4)}
    best = best | grid_result.best_params_
    for key in best:#trasformo eventuali liste in stringhe
        r=best[key]
        if type(r)==list:
            r=', '.join(map(str,r))
            best[key]=r
    best = pd.DataFrame(best, index=["Best"])
    #units risulta lista di liste
    for key in d:  # trasformo eventuali liste in stringhe
        print(d[key])
        for i in range(len(d[key])):
                if type(d[key][i])==list:
                    d[key][i] = ', '.join(map(str,d[key][i]))

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

def Wilcoxon(metrics,file='EvaluationCross/Wilcoxon',diag=True):
    for metric in metrics:
        data = pd.read_csv(f'{path}/EvaluationCross/EvaluationCrossSingle/{metric}.csv', index_col=[0])
        r = {}

        for i in range(len(data.index)):
            w = []
            model1 = data.index[i]
            val1 = data[data.index == model1].values[0]
            for model in data.index:
                if model != model1:  # and model not in Used:
                    val2 = data[data.index == model].values[0]
                    w.append(wilcoxon(val1, val2)[1])
                elif model == model1:
                    w.append(0)
            r[model1] = w  # modello:valori wil

        ris = pd.DataFrame(r, index=data.index)
        if file != '' and CheckPath(f'{path}/{file}'): # salva plot
            plt.style.use('seaborn-notebook')
            f, ax = plt.subplots(figsize=(10, 8))
            plt.title(f'Wilcoxon {metric}', fontdict={'family': 'Arial', 'color': '#001568', 'weight': 'normal', 'size': 20})
            if diag:
                ris = ris.drop(columns=ris.columns[-1])
                mask = np.triu((np.ones_like(ris, dtype=bool)))
                sns.heatmap(ris[1:], mask=mask[1:], cmap=sns.light_palette("seagreen", as_cmap=True,reverse=True), square=False, ax=ax, annot=True, fmt=".3f")
            else:
                 sns.heatmap(ris, mask=np.zeros_like(ris, dtype=np.bool), cmap=sns.light_palette("seagreen", as_cmap=True,reverse=True),square=True, ax=ax, annot=True, fmt=".3f")

            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=15, fontname='Arial')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15, fontname='Arial')
            plt.tight_layout()
            plt.savefig(f'{path}/{file}/Graph/{metric}')

            ris.to_csv(f'{path}/{file}/{metric}.csv', float_format="%.3f")
            with open(f'{path}/{file}/{metric}.txt', 'w') as f:
                f.write(ris.to_string(header=True, index=True, float_format="%.3f"))
        else:
            return ris

if __name__ == '__main__':
    x, y = PrepareData("JenaClimate.csv")
    print(y[:10])

