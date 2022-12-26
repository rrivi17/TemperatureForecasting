import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#tolgo info e warning
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestRegressor
from Plotting import Hist,PlotAllColumns,ScatterMatrix,CorrelationMatrix,ScatterPlot
import math
import sys
path=sys.path[1]



if __name__ == '__main__':
    df = pd.read_csv(f"{path}/PreProcessing/Dataset/JenaClimatePre.csv", index_col=[0])

    #fa scatter plot e plot distribuzioni
    #Hist(df['T (degC)'])

    CorrelationMatrix(df,file="PreProcessing/Graph/CorrelationMatrixDiagonal")
    CorrelationMatrix(df,diag=False,file="PreProcessing/Graph/CorrelationMatrix")

    ScatterMatrix(df,file="PreProcessing/Graph/ScatterMatrix")
    print(df.columns)
    for column in df.columns:
        if column!='T':
            ScatterPlot(x=df[column],y=df['T'],xlabel=column,ylabel="T",file=f"PreProcessing/Graph/ScatterPlot_{column}")
        else:
            continue

    PlotAllColumns(df,kind='line',file='PreProcessing/Graph/AllColumns2',title="All Columns Plot")
    PlotAllColumns(df,kind='scatter',target_name='T',file='PreProcessing/Graph/ScatterPlotAll')


    #print(df.describe())


