import pandas as pd
import numpy as np
import datetime as dt
from sklearn.impute import SimpleImputer
import sys
path=sys.path[1]

def CheckNan(df,Check=True):
    nan=df.isna().sum()
    for i in nan.index:
        if nan[i]>0:
            print(f'colonna {i} ha valori nan')
            Check=False
    return Check

def ManageNan(df,command):
    strategy={1:'mean',2:'median',3:'most_frequent'}
    if command<=3:#sostituisci con valori medi,mediani e più frequenti
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy[command])
        imputer = imputer.fit(df)
        df= imputer.transform(df)
    elif command==4:#sostituisci con 0
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=0)
        imputer = imputer.fit(df)
        df = imputer.transform(df)
    elif command==5:#cancella riga
        df.dropna(axis=0,inplace=True)
    elif command==6:#cancella colonna
        df.dropna(axis=1, inplace=True)
    else:
        df="errore"
    return df

if __name__ == '__main__':
    date_parser = lambda x: dt.datetime.strptime(x, "%d.%m.%Y %H:%M:%S")
    df = pd.read_csv(f"{path}/PreProcessing/Dataset/jena_climate_2009_2016.csv", index_col=['Date Time'], parse_dates=['Date Time'],date_parser=date_parser)
    df = df.drop("Tpot (K)",axis=1)#,"wv (m/s)","max. wv (m/s)","wd (deg)"])#tolgo temperatura espressa in kelvin e tengo solo quella in Celsius

    #ridefinizione nomi colonne
    newCol=[]
    for col in df.columns:
        if "(" in col:
            col=col.split('(')[0].strip().replace('. ','_')
            newCol.append(col)
    df.columns=newCol

    ok=CheckNan(df)#controllo presenza valori nan

    if not ok:
        command = int(input("-Digita 1 se li vuoi sostituiore con valori medi\n"
                            "-Digita 2 se li vuoi sostituire con valori mediani\n"
                            "-Digita 3 se li vuoi sostituire con valori più frequenti\n"
                            "-Digita 4 se li vuoi sostituire con 0\n"
                            "-Digita 5 se vuoi eliminare righe che hanno Nan\n"
                            "-Digita 6 se vuoi eliminare colonne che hanno Nan\n"))
        df=ManageNan(df,command)

    else:
        MaxT=df.groupby(df.index.date).max()['T']
        MinT=df.groupby(df.index.date).min()['T']

        df = df.groupby(df.index.date).mean()  # calcolo valori medi giornalieri
        df['TMax']=MaxT
        df['TMin']=MinT

        #elimino outliers su velocità venti
        df['max_wv'][df['max_wv'] < -200] = np.mean(df['max_wv'][df['max_wv'] > -200])
        df['wv'][df['wv'] < -200] = np.mean(df['wv'][df['wv'] > -200])

        #salvo file originale per attuare procedure di preprocessig/exploration
        df.to_csv(f"{path}/PreProcessing/Dataset/JenaClimatePre.csv", float_format="%.3f")

        #tolgo colonne identificate poco utili per la fase di preprocessing
        df=df.drop(columns=['wv','max_wv','wd'])
        df.to_csv(f"{path}/JenaClimate.csv", float_format="%.3f")