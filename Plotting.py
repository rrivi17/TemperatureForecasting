import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#tolgo info e warning
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.gridspec as gridspec
import datetime as dt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys
from tensorflow.keras.utils import plot_model
import math
from sklearn import tree
import pydotplus
from scipy.stats import wilcoxon

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
    plt.title('Residual Plot',fontdict={'family': 'Arial', 'color': '#001568', 'weight': 'normal', 'size': 20})
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

def LearningCurves(model, X, y,file=''):
 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
 train_errors, val_errors = [], []
 for m in range(1, len(X_train)):
    model.fit(X_train[:m], y_train[:m])
    y_train_predict = model.predict(X_train[:m])
    y_val_predict = model.predict(X_val)
    train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_val_predict))
 plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
 plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
 plt.yscale('log', base=10)
 plt.xscale('log', base=10)
 plt.legend()
 if file != '' and CheckPath(f'{path}/{file}'):
     plt.savefig(f'{path}/{file}')
 else:
     plt.show()

def ScatterMatrix(data,file=''):#non riesco a mettere titolo
    plt.style.use('seaborn-notebook')
    attributes = data.columns  # [lista colonne]
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
    #così stampo solo la metà della matrice, ok
    if diag:
        corr = corr.drop(columns=corr.columns[-1])
        mask=np.triu((np.ones_like(corr, dtype=bool)))
        sns.heatmap(corr[1:], mask=mask[1:], cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True,
                    fmt=".2f")
    else:
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, annot=True, fmt=".2f")
   # sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, annot=True,fmt=".2f")
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
    plt.title('Training and validation loss', fontdict={'family': 'Arial', 'color': '#001568', 'weight': 'normal', 'size': 20})
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
    axs = []

    m = nfigs % ncols
    m = range(1, ncols + 1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m * ncols)
    for i in range(0, nfigs):
        row = i // ncols
        col = i % ncols

        if row == nrows - 1:  # center only last row
            off = int(m * (ncols - nfigs % ncols) / 2)
        else:
            off = 0
        ax = plt.subplot(gs[row, m * col + off: m * (col + 1) + off])
        axs.append(ax)
    return axs
'''def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def PlotAllColumns(data,figsize=(12,8),col=3,file='',kind='line',title=''):
    plt.style.use('seaborn-notebook')

    row = math.ceil(len(data.columns) / col)
    #sharex='col'
    fig1, axs = plt.subplots(row, col, figsize=figsize, constrained_layout=True, tight_layout=True)

    if title != '':
        plt.suptitle(title, fontdict={'color': '#001568', 'weight': 'normal'},fontsize=20,fontname='Calibri Light')

    axs = trim_axs(axs, len(data.columns))
    data.plot(ax=axs, subplots=True, rot=20, title=list(data.columns),kind=kind,fontsize=10)
    plt.subplots_adjust()

    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()'''

def PlotAllColumns(data,target_name='',figsize=(12,8),col=3,file='',kind='line',title=''):#dà errore se numero andrebbe bene
    plt.style.use('seaborn-notebook')
    plt.figure()

    if len(data.columns) % col==0:
        row=int(len(data.columns) / col)
        fig1, axs = plt.subplots(row, col, figsize=figsize, constrained_layout=False, tight_layout=True)
    else:
        row = math.ceil(len(data.columns) / col)
        fig1, axs = plt.subplots(row, col, figsize=figsize, constrained_layout=False, tight_layout=True)
        axs = trim_axs(nrows=row, ncols=col, nfigs=len(data.columns))
    #fig1, axs = plt.subplots(row, col, figsize=figsize, constrained_layout=False, tight_layout=True)
    if title != '':
        plt.suptitle(title, fontdict={'color': '#001568', 'weight': 'normal'},fontsize=17,fontname='Arial')

    if target_name=='':
        bars = data.plot(ax=axs, subplots=True, rot=0, title=list(data.columns), kind=kind,fontsize=8)
    else:
        i=0
        for l in axs:
            try:
                for ax in l:
                    data.plot(ax=ax, rot=0, title=data.columns[i], kind=kind, fontsize=8,y=target_name,
                              x=data.columns[i],sharey=True,xlabel='',fontname='Arial',fontcolor='#001568')
                             # fontdict={'color': '#001568','fontfamily':'Arial'})
                    #ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=15, fontname='Arial')
                    #ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15, fontname='Arial')
                    i+=1
            except:
                data.plot(ax=l,  rot=0, title=data.columns[i], kind=kind, fontsize=8, y=target_name,
                          x=data.columns[i],sharey=True,xlabel='',fontname='Arial',fontcolor='#001568')
                i += 1

    plt.tight_layout()
    if file != '' and CheckPath(f'{path}/{file}'):
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()


def BarhSingle(data,folder=''):
    metrics = data.index
    models = data.columns
    for metric in metrics:
        plt.figure(figsize=(8, 4))
        plt.title(f'{metric}', fontdict={'family': 'Arial', 'color': '#001568', 'weight': 'normal', 'size': 15})
        data = data.sort_values(by=[f'{metric}'], axis=1)  # ordino valori in alto il più alto
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

def PlotAllWilcoxon(metrics,file='EvaluationCross/Wilcoxon/Graph/AllW',title='Wilcoxon',col=2,figsize=(15,10)):
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
        data = pd.read_csv(f'{path}/EvaluationCross/EvaluationCrossSingle/{metric}.csv', index_col=[0])
        r = {}
        n_ax+=1
        for i in range(len(data.index)):
            w = []
            model1 = data.index[i]
            val1 = data[data.index == model1].values[0]
            for model in data.index:
                if model != model1:
                    val2 = data[data.index == model].values[0]
                    w.append(wilcoxon(val1, val2)[1])
                elif model == model1:
                    w.append(0)
            r[model1] = w  # modello:valori wil

        ris = pd.DataFrame(r, index=data.index)
        ax = axs[n_ax]
        ax.set_title(f'{metric}',fontdict={'color': '#001568','fontfamily':'Arial'})

        ris = ris.drop(columns=ris.columns[-1])
        mask = np.triu((np.ones_like(ris, dtype=bool)))
        sns.heatmap(ris[1:], mask=mask[1:], cmap=sns.light_palette("seagreen", as_cmap=True,reverse=True), square=False, ax=ax,annot=True, fmt=".3f")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=12, fontname='Arial')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12, fontname='Arial')

    plt.subplots_adjust(left=0.145,bottom=0.16,wspace=1,hspace=1,top=0.9)

    if file != '' and CheckPath(f'{path}/{file}'):  # salva plot
        plt.savefig(f'{path}/{file}')
    else:
        plt.show()

