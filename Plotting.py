import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#tolgo info e warning
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.gridspec as gridspec
import datetime as dt
import sys
from tensorflow.keras.utils import plot_model
import math
from sklearn import tree
import pydotplus
from Function import ComputeTest

path=sys.path[1]
def CheckPath(file):
    if '/' not in file:
        return True
    else:#tolgo nome file
        path='/'.join(file.split('/')[:-1])
        isdir = os.path.isdir(path)
        return isdir

def PlotSplit(x,y,cv,file=''):
    plt.style.use('seaborn-notebook')
    plt.figure()
    plt.title("Time Series Split", fontdict={'family': 'Calibri', 'color': '#001568', 'weight': 'normal', 'size': 20})
    plt.yticks([n for n in range(1, cv.n_splits + 1)])
    plt.ylabel('CV')
    plt.xlabel('Date')
    i = 1
    for train, test in cv.split(x):
        plt.plot(y.index[train], [i for k in range(len(train))], color='blue')
        plt.plot(y.index[test], [i for k in range(len(test))], color='red')
        i += 1
    plt.legend(['Training set', 'Test set'])
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

def PlotTree(model,file=''):
    dot_data = tree.export_graphviz(model, filled=True)#esporto grafico in formato adatto
    pydot_graph = pydotplus.graph_from_dot_data(dot_data)
    if file != '' and CheckPath(f'{path}/{file}'):
        pydot_graph.write_png(f'{path}/{file}.png')
    else:
        plt.show()

def ScatterPlot(x,y,xlabel='y_test',ylabel='y_pred',file=''):
    plt.style.use('seaborn-notebook')
    plt.figure(figsize=(10, 6))
    #plt.title('Scatter Plot', fontdict={'family': 'Arial', 'color': '#001568', 'weight': 'normal', 'size': 20})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim((min(y)-5,max(y) + 10))
    sns.scatterplot(x=x,y=y)
    plt.tight_layout()
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

def ScatterPlotOutliers(x,y,xlabel='y_test',ylabel='y_pred',threshold=20,file=''):
    plt.style.use('seaborn-notebook')
    plt.figure(figsize=(10, 6))
    plt.title('Scatter Plot', fontdict={'family': 'Arial', 'color': '#001568', 'weight': 'normal', 'size': 20})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    condition=abs(x-y)>threshold
    is_outlier=np.where(condition,'outlier','non-outlier')
    sns.scatterplot(x=x,y=y, hue=is_outlier)
    plt.tight_layout()
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

def ResidualPlot(y_test,y_pred,file=''):
    plt.style.use('seaborn-notebook')
    ris={'y_test':y_test,'y_pred':y_pred}
    ris=pd.DataFrame(ris)

    plt.figure(figsize=(10, 6))
    #plt.title('Residual Plot',fontdict={'family': 'Arial', 'color': '#001568', 'weight': 'normal', 'size': 20})
    sns.residplot(x=ris['y_test'], y=ris['y_pred'])

    plt.tight_layout()
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

def PlotModel(model,file=''):
    if file != '' and CheckPath(f'{path}/{file}'):
        plot_model(model, show_shapes=True, to_file=f'{path}/{file}')


def PlotLoss(history,file=''):
    plt.style.use('seaborn-notebook')
    plt.figure(figsize=(8, 4))
    # loss=model.history.history['loss']
    plt.plot(history.history['loss'],color='#001568')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log', base=10)

    plt.tight_layout()
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

def Hist(column, file=''):
    plt.style.use('seaborn-notebook')
    plt.figure(figsize=(10,8))
    plt.title(f'Hist of {column.name}',fontsize=20,color='#990000')
    plt.hist(column,bins=50,color='#FF9933')
    print(f'{path}/{file}')
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

def ScatterMatrix(data,file=''):
    plt.style.use('seaborn-notebook')
    attributes = data.columns
    axes = pd.plotting.scatter_matrix(data[attributes], figsize=(12, 8))
    #pd.plotting.scatter_matrix(data[attributes], figsize=(12, 8))
    plt.suptitle('Scatter Matrix',fontsize=20,color='#001568')
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

def CorrelationMatrix(data,file='',method='pearson',diag=True):
    #alternativa method 'spearman'
    plt.style.use('seaborn-notebook')
    f, ax = plt.subplots(figsize=(10, 8))
    corr = data.corr(method=method)
    if diag:
        corr = corr.drop(columns=corr.columns[-1])
        mask=np.triu((np.ones_like(corr, dtype=bool)))
        sns.heatmap(corr[1:], mask=mask[1:], cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True,
                    fmt=".2f")
    else:
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, annot=True, fmt=".2f")
    #ax.set_title('Correlation Matrix', fontsize=20, color='#001568')
    plt.tight_layout()

    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()


def TrainValAcc(history,file=''):
    plt.style.use('seaborn-notebook')
    plt.figure(figsize=(10, 7), facecolor='#FFFFCC')
    plt.axes().set_facecolor('#F8F8FF')
    plt.title('Training and validation accuracy',fontdict={'family': 'Arial', 'color': 'darkred', 'weight': 'normal', 'size': 20,'style': 'italic'})
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)

    loss=history.history['loss']
    epochs = range(1, len(loss) + 1)
    if 'accuracy' in history.history.keys():
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
    else:
        acc=history.history['acc']
        val_acc = history.history['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.legend()
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()


def TrainValLoss(history,file=''):
    plt.style.use('seaborn-notebook')
    plt.figure(figsize=(10, 7))# facecolor='#FFFFCC')
    #plt.axes().set_facecolor('#F8F8FF')
    #plt.title('Training and validation loss', fontdict={'family': 'Arial', 'color': '#001568', 'weight': 'normal', 'size': 20})
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss',color='#001568')
    plt.plot(epochs, val_loss, label='Validation loss',color='red')

    plt.yscale('log', base=10)

    plt.tight_layout()
    plt.legend()
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

def trim_axs(nrows,ncols,nfigs):
    axs=[]

    m=nfigs%ncols
    m=range(1, ncols +1)[-m]
    gs=gridspec.GridSpec(nrows, m*ncols)
    for i in range(0, nfigs):
        row=i // ncols
        col=i%ncols

        if row == nrows-1: #accentro ultima colonna
            off=int(m*(ncols-nfigs%ncols)/2)
        else:
            off=0
        ax = plt.subplot(gs[row, m*col+off: m*(col+1)+off])
        axs.append(ax)
    return axs

def PlotAllColumns(data,target_name='',figsize=(12,8),col=3,file='',kind='line',title=''):
    plt.style.use('seaborn-notebook')
    if len(data.columns)%col==0:
        row=int(len(data.columns) / col)
        fig1, axs = plt.subplots(row, col, figsize=figsize, constrained_layout=False, tight_layout=True)
    else:
        row = math.ceil(len(data.columns) / col)
        fig1, axs = plt.subplots(row, col, figsize=figsize, constrained_layout=False, tight_layout=True)
        axs = trim_axs(nrows=row, ncols=col, nfigs=len(data.columns))

    if title != '':
        plt.suptitle(title, fontdict={'color': '#001568', 'weight': 'normal'},fontsize=17,fontname='Arial')

    if target_name=='' and kind!='barh':
        colonne=list(data.columns)
        for i in range(len(colonne)):
            if colonne[i]=='R2':
                colonne[i]='$R^2$'
        data.plot(ax=axs, subplots=True, rot=0, title=colonne, kind=kind, fontsize=10, legend=False)

    else:
        i=0
        for l in axs:
            if data.columns[i] == 'R2':
                title = '$R^2$'
            else:
                title=data.columns[i]
            try:
                for ax in l:
                    if kind=='barh' and target_name =='':
                        bars = ax.barh(width=data[data.columns[i]], y=data.index)
                        ax.bar_label(bars, label_type='center', fontsize=15, fontname='Arial', color='white')
                        ax.set_title(title)

                    elif target_name != '':
                        data.plot(ax=ax, rot=0, title=title, kind=kind, fontsize=8,y=target_name,
                                      x=data.columns[i],sharey=True,xlabel='',legend=False)#fontname='Arial',fontcolor='#001568'
                    i+=1

            except:
                if kind=='barh' and target_name == '':
                    bars = l.barh(width=data[data.columns[i]], y=data.index)
                    l.bar_label(bars, label_type='center', fontsize=15, fontname='Arial', color='white')
                    l.set_title(title)

                elif target_name != '':
                    data.plot(ax=l,  rot=0, title=title, kind=kind, fontsize=8, y=target_name,legend=False,
                                  x=data.columns[i],sharey=True,xlabel='')#fontname='Arial',fontcolor='#001568'
                i += 1

    plt.tight_layout()
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()


def Barh(data,folder=''):
    metrics = data.index

    for metric in metrics:
        plt.figure(figsize=(8, 4))
        #fig,ax=plt.subplots(figsize=(8, 4))
        if metric == 'R2':
            title = '$R^2$'
        else:
            title = metric

        plt.title(f'{title}', fontdict={'family': 'Arial', 'color': '#001568', 'weight': 'normal', 'size': 15})
        data = data.sort_values(by=[f'{metric}'], axis=1)
        models = data.columns
        bars = plt.barh(models, width=data.loc[metric])
        plt.bar_label(bars, label_type='center', fontsize=15, fontname='Arial', color='white')

        plt.tight_layout()
        if folder != '' and CheckPath(f'{path}/{folder}'):
            plt.savefig(f'{path}/{folder}/{metric}')
        else:
            plt.show()

def Graph(y_test,yp_test,file=''):
    plt.clf()
    plt.style.use('seaborn-notebook')
    #plt.figure(figsize=(10, 7), facecolor='#FFFFCC')
    plt.figure(figsize=(12, 8))
    #plt.axes().set_facecolor('#F8F8FF')
    plt.title('Daily Temperature Prediction', fontdict={'family': 'Arial', 'color': '#001568','weight': 'normal', 'size': 20})
    plt.xlabel("Date", fontsize=15)
    plt.ylabel('Temperature', fontsize=15)
    #plt.xticks(rotation=45, ha='right')
    #plt.plot(y_train.index, y_train.values, label='Train')
    plt.plot(y_test.index, y_test.values, label='True')
    plt.plot(y_test.index, yp_test, label='Prediction')
    plt.tight_layout()
    plt.legend()

    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

def PlotAllWilcoxon(metrics,file='',title='',col=2,figsize=(17,10)):
    if len(metrics) % col==0:
        row=int( len(metrics) / col)
        fig1, axs = plt.subplots(row, col, figsize=figsize, constrained_layout=False, tight_layout=True)
    else:
        row = math.ceil( len(metrics) / col)
        fig1, axs = plt.subplots(row, col, figsize=figsize, constrained_layout=False, tight_layout=True)
        axs = trim_axs(nrows=row, ncols=col, nfigs=len(metrics))
    if title != '':
        plt.suptitle(title, fontdict={'color': '#001568', 'weight': 'normal'}, fontsize=17, fontname='Arial')


    n_ax=-1
    for metric in metrics:
        data = pd.read_csv(f'{path}/EvaluationCross/EvaluationCross/{metric}.csv', index_col=[0])
        n_ax += 1
        ris = ComputeTest(data, test='wilcoxon')

        ax = axs[n_ax]
        if metric == 'R2':
            title = '$R^2$'
        else:
            title = metric
        ax.set_title(f'{title}',fontdict={'color': '#001568','fontfamily':'Arial'})

        ris = ris.drop(columns=ris.columns[-1])
        mask = np.triu((np.ones_like(ris, dtype=bool)))
        sns.heatmap(ris[1:], mask=mask[1:], cmap=sns.light_palette("seagreen", as_cmap=True,reverse=True), square=False, ax=ax,annot=True, fmt=".3f")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=15, fontname='Arial')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15, fontname='Arial')

    plt.subplots_adjust(left=0.145,bottom=0.16,wspace=1,hspace=1,top=0.9)

    if file != '' and CheckPath(f'{path}/{file}'):  # salva plot
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

def PlotAllTtest(metrics,file='',title='',col=2,figsize=(17,10)):
    if len(metrics) % col==0:
        row=int( len(metrics) / col)
        fig1, axs = plt.subplots(row, col, figsize=figsize, constrained_layout=False, tight_layout=True)
    else:
        row = math.ceil( len(metrics) / col)
        fig1, axs = plt.subplots(row, col, figsize=figsize, constrained_layout=False, tight_layout=True)
        axs = trim_axs(nrows=row, ncols=col, nfigs=len(metrics))
    if title != '':
        plt.suptitle(title, fontdict={'color': '#001568', 'weight': 'normal'}, fontsize=17, fontname='Arial')

    n_ax=-1
    for metric in metrics:
        data = pd.read_csv(f'{path}/EvaluationCross/EvaluationCross/{metric}.csv', index_col=[0])
        n_ax+=1
        ris = ComputeTest(data, test='Ttest')

        ax = axs[n_ax]
        if metric == 'R2':
            title = '$R^2$'
        else:
            title = metric
        ax.set_title(f'{title}', fontdict={'color': '#001568', 'fontfamily': 'Arial'})

        ris = ris.drop(columns=ris.columns[-1])
        mask = np.triu((np.ones_like(ris, dtype=bool)))
        sns.heatmap(ris[1:], mask=mask[1:], cmap=sns.light_palette("seagreen", as_cmap=True, reverse=True),
                    square=False, ax=ax, annot=True, fmt=".3f")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=15, fontname='Arial')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15, fontname='Arial')

    plt.subplots_adjust(left=0.145, bottom=0.16, wspace=1, hspace=1, top=0.9)

    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

