import pandas as pd
from Plotting import PlotAllColumns,PlotAllWilcoxon,PlotAllTtest
from Function import Wilcoxon,Ttest
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
path=sys.path[1]
import os

def CheckPath(file):
    if '/' not in file:
        return True
    else:#tolgo nome file
        path='/'.join(file.split('/')[:-1])
        isdir = os.path.isdir(path)
        return isdir


def CompareMetric(metric,test='Ttest',file=''):
    data = pd.read_csv(f'{path}/EvaluationCross/{test}/{metric}.csv', index_col=[0])
    m=pd.read_csv(f'{path}/EvaluationCross/EvaluationCrossMean.csv', index_col=[0])
    m=m.loc[metric]

    ris = {}

    for model in data.index:
        for model2 in data.index:
            if model not in ris:
                ris[model]=[]
            if data[model].loc[model2]<0.05:#Ttest<0.05
                ris[model].append(round(m[model]-m[model2],3))
            else:
                ris[model].append(0)


    ris=pd.DataFrame(ris,index=data.index)
    labels=ris.values.astype(str)
    labels[np.where(labels=='0.0')]='-'

    fig,ax=plt.subplots(figsize=(8,6))
    ax.set_title(metric+'$_{x}$'+' - '+metric+'$_{y}$',fontdict={'family': 'Arial', 'color': 'black', 'weight': 'normal', 'size': 20})
    sns.heatmap(ris, mask=np.zeros_like(ris, dtype=bool), cmap=sns.color_palette("vlag", as_cmap=True),
                square=True, ax=ax,  fmt="",annot=labels)

    ax.set_ylabel(metric+'$_{y}$',fontdict={'family': 'Arial', 'color': 'black', 'weight': 'normal', 'size': 15})
    ax.set_xlabel(metric+'$_{x}$',fontdict={'family': 'Arial', 'color': 'black', 'weight': 'normal', 'size': 15})

    plt.tight_layout()
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()


if __name__ == '__main__':
    #analisi metriche con hold-out method
    data = pd.read_csv(f'{path}/Evaluation/Evaluation.csv', index_col=[0])
    PlotAllColumns(data.T,kind='barh',file='Evaluation/Graph/AllMetrics',col=2)
    metrics = data.index
    models=data.columns

    #analisi metriche con cross validation method
    dataCross = pd.read_csv(f'{path}/EvaluationCross/EvaluationCrossMean.csv', index_col=[0])
    PlotAllColumns(dataCross.T, kind='barh', file='EvaluationCross/Graph/AllMetricsCross', col=2)

    #wilcoxon
    Wilcoxon(metrics)
    PlotAllWilcoxon(metrics)

    #ttest
    Ttest(metrics)
    PlotAllTtest(metrics)

    #ris=pd.read_csv(f'{path}/EvaluationCross/Ttest/R2.csv', index_col=[0])

    CompareMetric('R2',test='Ttest',file="EvaluationCross/Ttest/Graph/CompareR2")
