import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Function import PrepareData,SaveGridSearch,Scaler3D
from sklearn.model_selection import TimeSeriesSplit,RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.pipeline import Pipeline


def GridSearch(x, y):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8, restore_best_weights=True)
    model = KerasRegressor(model=build_model,units=[20,5,10],n_layers=1,activation='relu',callbacks=es,delta=1,recurrent_activation='sigmoid',validation_split=0.2)

    # PARAMETRI
    delta=[1,1.35,0.8]
    activation=['relu','tanh','selu']
    optimizer = ['adam', 'nadam']
    recurrent_activation=['sigmoid','selu','relu']
    batch_size = [32,25]
    epochs = [30]
    units = []
    #dropout=[]
    n_layers = [2,3,4]

    for size in  n_layers:  # calcolo in modo random units da assegnare a ogni layer
        for i in range(0,30):
            units.append((np.random.randint(2,9, size=size)*20).tolist())
            #dropout.append(((np.random.randint(1, 3, size) -1)/ 10).tolist())


    param_grid = []
    for n in set(n_layers):
        param = {'GRU__n_layers': [n]}
        #param['dropout'] = [d for d in dropout if len(d) == n]
        param['GRU__units'] = [u for u in units if len(u) == n]
        param['GRU__epochs'] = epochs
        param['GRU__delta'] = delta
        param['GRU__activation'] = activation
        param['GRU__batch_size']=batch_size
        param['GRU__optimizer']=optimizer
        param['GRU__recurrent_activation']=recurrent_activation
        param_grid.append(param)


    model = Pipeline(steps=[('preprocessing', Scaler3D()), ('GRU', model)])
    TimeSplit = TimeSeriesSplit(n_splits=5)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=4, cv=TimeSplit,scoring='r2',n_iter=30) #scoring=['neg_mean_squared_error','r2'],refit=False)

    grid_result = grid.fit(x, y)
    SaveGridSearch(param_grid,grid_result,folder='GRU/GridSearch')


if __name__ == '__main__':
    x, y = PrepareData("JenaClimate.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=False)

    input_shape = (x_train.shape[1], x_train.shape[2])


    def build_model(input_shape=input_shape,units=[20,5,10],epochs=2, batch_size=25,n_layers=1,activation='relu',delta=1,recurrent_activation='sigmoid',optimizer='adam'):

        model = Sequential()

        for i in range(n_layers):
            if i==0:
                model.add(GRU(units=units[i], return_sequences=True, input_shape=input_shape,recurrent_activation=recurrent_activation))
            elif i==n_layers-1:
                model.add(GRU(units=units[i],recurrent_activation=recurrent_activation,activation=activation))
            else:
                model.add(GRU(units=units[i],return_sequences=True,activation=activation,recurrent_activation=recurrent_activation))


        model.add(Dense(units=1))

        loss = tf.keras.losses.Huber(delta=delta)
        model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.MeanSquaredError()])

        return model


    GridSearch(x_train, y_train)