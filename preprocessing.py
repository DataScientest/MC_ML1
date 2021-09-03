import pandas as pd 
import numpy as np 

def feature(df):
    """
    Création des variables d'historique des prix, des moyennes mobiles, volatilité
    """

    # Historique des prix 

    P= range(1,11)

    for p in P:
        df['Open_'+ str(p)]=df['Open'].shift(p)
        
    #df.head()

    # Moyennes mobiles 

    M=[5,10,15,20]

    for m in M:
        df['MA_'+str(m)]=df['Open'].rolling(m).mean()
        

    # Retours sur investissement 

    R=[1,5,10,15]

    for r in R:
        df['R_'+str(r)]=df['Open'].pct_change(r)
        
    # Volatilité 

    V=[5,10,15]

    for v in V:
        df['V_'+str(v)]=df['R_1'].rolling(v).std()

    df['target']=df['V_5'].shift(-5)

    df['target']=df['target'].apply(lambda x : 1 if x>0.011 else 0)

    df=df.dropna()

    return df 

def split(df): 
    X=df.drop('target',axis=1)
    y=df.target

    X_train=X.iloc[:int(len(X)*0.7)]
    y_train=y.iloc[:int(len(y)*0.7)]

    X_test=X.iloc[int(len(X)*0.7):]
    y_test=y.iloc[int(len(y)*0.7):]

    return X_train,X_test,y_train,y_test 

