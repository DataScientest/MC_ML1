from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from joblib import load

def drop_columns(df):
    to_drop=['Date','High','Low','Close','Volume','OpenInt']
    
    return df.drop(to_drop,axis=1)


def make_pipeline(scaler_path,model_path):
    """
    Créer la pipeline composée de drop columns, normalisation et de modélisation
    """
    col_dropper=FunctionTransformer(drop_columns)

    scaler=load(scaler_path)

    model=load(model_path)

    return Pipeline([
        ("drop_columns",col_dropper),
        ("scaling",scaler),
        ("model",model)
    ])

