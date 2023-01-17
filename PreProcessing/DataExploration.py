import pandas as pd
from Plotting import PlotAllColumns,ScatterMatrix,CorrelationMatrix,ScatterPlot
import sys
path=sys.path[1]


if __name__ == '__main__':
    df = pd.read_csv(f"{path}/PreProcessing/Dataset/JenaClimatePre.csv", index_col=[0])

    CorrelationMatrix(df,file="PreProcessing/Graph/CorrelationMatrixDiagonal")
    CorrelationMatrix(df,diag=False,file="PreProcessing/Graph/CorrelationMatrix")

    ScatterMatrix(df,file="PreProcessing/Graph/ScatterMatrix")

    for column in df.columns:
        if column!='T':
            ScatterPlot(x=df[column],y=df['T'],xlabel=column,ylabel="T",file=f"PreProcessing/Graph/ScatterPlot_{column}")
        else:
            continue

    PlotAllColumns(df,kind='line',file='PreProcessing/Graph/AllColumns',title="All Columns Plot")
    PlotAllColumns(df,kind='scatter',target_name='T',file='PreProcessing/Graph/ScatterPlotAll')



