import numpy as np
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder

import warnings;warnings.filterwarnings("ignore")

from itertools import groupby

def label_encoder(series, fill_value = ""):
    """
    """
    uniques = sorted(series.fillna(fill_value).unique())
    label = range(len(uniques))
    mapping = dict(zip(uniques, label))
    return series.map(mapping)


def train_binary(frame,
                 features,
                 target_column,
                 estimator,
                 cv = 3,
                 refit_all = True,
                 verbose = True):
    """
    Creates a feature based on predictions from kfold
    """
    ls = list()
    result = {}
    kf = StratifiedKFold(n_splits = cv)
    k = 0
    
    X = frame[features]
    y = frame[target_column]

    encoder = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), selector(dtype_include="object"))],
    remainder="passthrough"
                            )


    clf = Pipeline([("preprocessor",encoder), ("estimator",estimator)])

    for train_idx, test_idx in kf.split(X, y):
        k+=1
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = clf.fit(X_train, y_train)
        predictions = clf.predict_proba(X_test)[:,1]
        if verbose:
            score = roc_auc_score(y_true=y_test, y_score=predictions)
            print(f"Score on test set for fold {k} is :{round(score,3)}")
        
        ls.append(predictions)
        
    feature = np.hstack(ls)
    
    if refit_all:
        model = clf.fit(X, y)
        
        result["model"] = model
    
    result["data"] = frame.assign(**{f"prediction":feature})

    def p(new_df):
        return new_df.assign(**{f"prediction":model.predict_proba(new_df[features])[:,1]})

    result["p"] = p
    
    return result

def get_entropy(series, categorical = True):

    try:
        if categorical:
            data = series.value_counts(True)
        else:
            data = pd.qcut(series,[0, .25, .5, .75, 1.], duplicates= "drop").value_counts(True)

        return entropy(data)
    except:
        return 0

def len_iter(items):
    return sum(1 for _ in items)


def consecutive_values(data, bin_val):
    try:
        return max(len_iter(run) for val, run in groupby(data) if val == bin_val)/len(data)
    except:
        return 0

def get_linear_regression(series):
    result ={}
    try:
        n = len(series)
        X = np.arange(n).reshape(-1,1)
        y = series.cumsum()
        lr = LinearRegression.fit(X, y)
    
        result["trend"] = lr.slope
        result["intercept"] = lr.intercept
    except:
        result["trend"] = -1
        result["intercept"] = -1
    return result


def find_constraint(frame, feature, target, q:int = 10):
    """
    """
    try:
        lr = LinearRegression()
        cuts = pd.qcut(frame[feature], q, duplicates = "drop")
    
        df = frame.groupby(cuts)[target].mean().sort_index()
        X = np.arange(len(df)).reshape(-1,1) 
        
    
        lr.fit(X, df.values)
    
        return np.sign(lr.coef_[0])
    except:
        return 0


def risk_band_table(frame,
                    target_column:str,
                    prediction_column:str,
                    strategy:str = "tree",
                    n_bins:int = 10,
                    tree_params:dict = {}
                   ):
    """
    """
    if strategy == "tree":
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(**tree_params).fit(frame[predict_column], frame[target_column])
        cuts = clf.apply(frame[predict_column])

        def p(new_df):
            return new_df.assign(**{"risk_tier":clf.apply(new_df)})
    else:
        binarizer = KBinsDiscretizer(strategy = strategy, n_bins=n_bins,
                                encode="ordinal").fit(frame[prediction_column])
        
        cuts = np.ravel(binarizer.transform(frame[prediction_column]))

    tabla = frame.groupby(cuts).agg({"prediction": ["min","max"],
                          target_column:["sum","mean","count"]}
                        )
    tabla.columns = tabla.columns.map(' '.join)
    
    tabla = tabla.rename(columns = {"target sum":"events",
                        "target mean":"average risk",
                        "target count":"number of samples"})

    tabla["percentage sample"] = tabla["number of samples"]/tabla["number of samples"].sum()
    tabla["non events"] = tabla["number of samples"] - tabla["events"]
    tabla["cum event rate"] = (tabla["events"]/tabla["events"].sum()).cumsum()
    tabla["cum non event rate"] = (tabla["non events"]/tabla["non events"].sum()).cumsum()

    tabla["cum difference"] = np.abs(tabla["cum event rate"] - tabla["cum non event rate"])

    if strategy == "tree":
        return p, tabla
    else:
        return binarizer.bin_edges_, tabla


def load_csv_files(directory):
    """
    Load all .csv files from a directory and store them in a dictionary.

    Parameters:
        directory (str): Path to the directory containing .csv files.

    Returns:
        dict: A dictionary where keys are file names (without extension) and values are DataFrames.
    """
    data_dict = {}  # Dictionary to store loaded DataFrames

    # List all files in the directory
    file_list = os.listdir(directory)

    # Iterate through files and load .csv files
    for filename in file_list:
        try:
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)
                # import pdb;pdb.set_trace()
                # Remove file extension from the key
                key = os.path.splitext(filename)[0].lower()
                data_dict[key] = df
        except:
            pass
            
            

    return data_dict
