import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#tolgo info e warning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Function import PrepareData
from Plotting import PlotModel,ResidualPlot,Graph,ScatterPlot,PlotLoss,TrainValLoss
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping
from Function import SaveModel,EvaluationCross,Evaluation,Scaler3D
from scikeras.wrappers import KerasRegressor
from sklearn.pipeline import Pipeline


def build_model(input_shape,recurrent_activation='selu',optimizer='adam',activation='selu',delta=1.35):
    model = Sequential()

    model.add(GRU(units=160, return_sequences=True, input_shape=input_shape,recurrent_activation=recurrent_activation,activation=activation))
    model.add(GRU(units=120,activation=activation,recurrent_activation=recurrent_activation))

    model.add(Dense(units=1))

    model.summary()

    loss = tf.keras.losses.Huber(delta=delta)
    model.compile(loss=loss, optimizer=optimizer, metrics=['mean_absolute_error'])

    return model


def fit_model(x_train,y_train,epochs=100, batch_size=25):#25,32
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)#8
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,verbose=1, callbacks=[es],validation_split=0.2,shuffle=False)

    SaveModel(model,model_name="GRU")
    return history



if __name__ == '__main__':
    x, y = PrepareData("JenaClimate.csv")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2, shuffle=False)

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

    input_shape = (x_train.shape[1], x_train.shape[2])  # n sequenza, n colonne

    model = build_model(input_shape)
    PlotModel(model,file="GRU/Graph/GRUModel.png")#salvo struttura modello


    history = fit_model(x_train, y_train)
    PlotLoss(history,file='GRU/Graph/GRULoss')
    TrainValLoss(history,file='GRU/Graph/GRUTrainValLoss')

    y_pred = model.predict(x_test)
    y_pred=y_pred.ravel()


    Evaluation( y_test, y_pred, model_name='GRU', file="GRU/Evaluation/GRUEvaluation")
    ResidualPlot(y_test,y_pred,file="GRU/Graph/GRUResidualPlot")
    Graph(y_test, y_pred, file="GRU/Graph/GRURealVSPred")
    ScatterPlot(x=y_test, y=y_pred, file="GRU/Graph/GRUScatterPlot")

    input_shape = (x.shape[1], x.shape[2])


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)

    model=KerasRegressor(model=build_model, input_shape=input_shape, callbacks=es, batch_size=25, epochs=100, validation_split=0.2)
    model = Pipeline(steps=[('preprocessing', Scaler3D()), ('GRU', model)])

    EvaluationCross(model,x,y,model_name='GRU',file="GRU/Evaluation/GRUEvaluationCross",fileMean="GRU/Evaluation/GRUEvaluationCrossMean")
