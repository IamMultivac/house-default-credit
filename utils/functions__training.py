from typing import List, Dict,  Any, Optional, Callable, Tuple, Union, TYPE_CHECKING
from toolz import curry, merge, assoc
from itertools import chain, repeat, starmap

import cloudpickle as cp

from toolz.curried import curry, partial, pipe, assoc, accumulate, map, filter

from datetime import datetime, timedelta
import operator

from time import time

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn import __version__ as sk_version
import sys

# from fklearn.types import LearnerReturnType, LogType
# from fklearn.common_docstrings import learner_return_docstring, learner_pred_fn_docstring
# from fklearn.training.utils import log_learner_time, expand_features_encoded

sys.path.append("../")
from src.config import logger
from utils.fklearn__types import *
from utils.fklearn__training_utils import *


if TYPE_CHECKING:
    from lightgbm import Booster


# Log types
LogType = Dict[str, any]


# Learner types
# PredictFnType = Callable[[pd.DataFrame], pd.DataFrame]
# LearnerLogType = Dict[str, any]
# LearnerReturnType = Tuple[PredictFnType, pd.DataFrame, LearnerLogType]

@curry
@log_learner_time(learner_name='lgbm_classification_learner')
def lgbm_classification_learner(df: pd.DataFrame,
                                features: List[str],
                                target: str,
                                learning_rate: float = 0.1,
                                num_estimators: int = 100,
                                extra_params: Optional[LogType] = None,
                                prediction_column: str = "prediction",
                                weight_column: Optional[str] = None,
                                encode_extra_cols: bool = True,
                                valid_sets: Optional[List[pd.DataFrame]] = None,
                                valid_names: Optional[List[str]] = None,
                                feval: Optional[Union[
                                    Callable[[np.ndarray, pd.DataFrame], Tuple[str, float, bool]],
                                    List[Callable[[np.ndarray, pd.DataFrame], Tuple[str, float, bool]]]]
                                ] = None,
                                linear_trees:bool = False,
                                init_model: Optional[Union[str, Path, 'Booster']] = None,
                                feature_name: Union[List[str], str] = 'auto',
                                categorical_feature: Union[List[str], List[int], str] = 'auto',
                                keep_training_booster: bool = False,
                                callbacks: Optional[List[Callable]] = None,
                                dataset_init_score: Optional[Union[
                                    List, List[List], np.ndarray, pd.Series, pd.DataFrame]
                                ] = None) -> LearnerReturnType:
    """
    Fits an LGBM classifier to the dataset.

    It first generates a Dataset
    with the specified features and labels from `df`. Then, it fits a LGBM
    model to this Dataset. Return the predict function for the model and the
    predictions for the input dataset.

    Parameters
    ----------

    df : pandas.DataFrame
       A pandas DataFrame with features and target columns.
       The model will be trained to predict the target column
       from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.
        This column should be discrete, since this is a classification model.

    learning_rate : float
        Float in the range (0, 1]
        Step size shrinkage used in update to prevents overfitting. After each boosting step,
        we can directly get the weights of new features. and eta actually shrinks the
        feature weights to make the boosting process more conservative.
        See the learning_rate hyper-parameter in:
        https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

    num_estimators : int
        Int in the range (0, inf)
        Number of boosted trees to fit.
        See the num_iterations hyper-parameter in:
        https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

    extra_params : dict, optional
        Dictionary in the format {"hyperparameter_name" : hyperparameter_value}.
        Other parameters for the LGBM model. See the list in:
        https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
        If not passed, the default will be used.

    prediction_column : str
        The name of the column with the predictions from the model.

    weight_column : str, optional
        The name of the column with scores to weight the data.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.

    valid_sets : list of pandas.DataFrame, optional (default=None)
        A list of datasets to be used for early-stopping during training.

    valid_names : list of strings, optional (default=None)
        A list of dataset names matching the list of datasets provided through the ``valid_sets`` parameter.

    feval : callable, list of callable, or None, optional (default=None)
        Customized evaluation function. Each evaluation function should accept two parameters: preds, eval_data, and
        return (eval_name, eval_result, is_higher_better) or list of such tuples.

    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.

    feature_name : list of str, or 'auto', optional (default="auto")
        Feature names. If ‘auto’ and data is pandas DataFrame, data columns names are used.

    categorical_feature : list of str or int, or 'auto', optional (default="auto")
        Categorical features. If list of int, interpreted as indices. If list of str, interpreted as feature names (need
        to specify feature_name as well). If ‘auto’ and data is pandas DataFrame, pandas unordered categorical columns
        are used. All values in categorical features will be cast to int32 and thus should be less than int32 max value
        (2147483647). Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values. The output cannot be
        monotonically constrained with respect to a categorical feature. Floating point numbers in categorical features
        will be rounded towards 0.

    keep_training_booster : bool, optional (default=False)
        Whether the returned Booster will be used to keep training. If False, the returned value will be converted into
        _InnerPredictor before returning. This means you won’t be able to use eval, eval_train or eval_valid methods of
        the returned Booster. When your model is very large and cause the memory error, you can try to set this param to
        True to avoid the model conversion performed during the internal call of model_to_string. You can still use
        _InnerPredictor as init_model for future continue training.

    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration. See Callbacks in LightGBM Python API for more
        information.

    dataset_init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for
        multi-class task), or None, optional (default=None)
        Init score for Dataset. It could be the prediction of the majority class or a prediction from any other model.
    """

    import lightgbm as lgbm

    params = extra_params if extra_params else {}
    params = assoc(params, "eta", learning_rate)
    params = params if "objective" in params else assoc(params, "objective", 'binary')

    weights = df[weight_column].values if weight_column else None

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    dtrain = lgbm.Dataset(df[features],
                          label=df[target],
                          feature_name=list(map(str, features)),
                          weight=weights,
                          init_score=dataset_init_score,
                          params = {"linear_trees":linear_trees,
                                    "verbose":False}
                         )

    bst = lgbm.train(params=params,
                     train_set=dtrain,
                     num_boost_round=num_estimators,
                     valid_sets=valid_sets,
                     valid_names=valid_names,
                     feval=feval,
                     init_model=init_model,
                     feature_name=feature_name,
                     categorical_feature=categorical_feature,
                     keep_training_booster=keep_training_booster,
                     callbacks=callbacks
                    )

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        if params["objective"] == "multiclass":
            col_dict = {prediction_column + "_" + str(key): value
                        for (key, value) in enumerate(bst.predict(new_df[features]).T)}
        else:
            col_dict = {prediction_column: bst.predict(new_df[features])}

        if apply_shap:
            import shap
            explainer = shap.TreeExplainer(bst)
            shap_values = explainer.shap_values(new_df[features])
            shap_expected_value = explainer.expected_value

            if params["objective"] == "multiclass":
                shap_values_multiclass = {f"shap_values_{class_index}": list(value)
                                          for (class_index, value) in enumerate(shap_values)}
                shap_expected_value_multiclass = {
                    f"shap_expected_value_{class_index}":
                        np.repeat(expected_value, len(class_shap_values))
                    for (class_index, (expected_value, class_shap_values))
                    in enumerate(zip(shap_expected_value, shap_values))
                }
                shap_output = merge(shap_values_multiclass, shap_expected_value_multiclass)

            else:
                shap_values = list(shap_values[1])
                shap_output = {"shap_values": shap_values,
                               "shap_expected_value": np.repeat(shap_expected_value[1], len(shap_values))}

            col_dict = merge(col_dict, shap_output)

        return new_df.assign(**col_dict)

    log = {'lgbm_classification_learner': {
        'features': features,
        'target': target,
        'prediction_column': prediction_column,
        'package': "lightgbm",
        'package_version': lgbm.__version__,
        'parameters': assoc(params, "num_estimators", num_estimators),
        'feature_importance': dict(zip(features, bst.feature_importance().tolist())),
        'training_samples': len(df)},
        'object': bst}

    return p, p(df), log



def model_pipeline(train_df:pd.DataFrame,
                   validation_df:pd.DataFrame,
                   params:dict,
                   target_column:str,
                   features:list = None,
                   cv:int = 3,
                   random_state:int = 42,
                   apply_shap:bool = False,
                   save_estimator_path:str = None
                  ):
    """

    """
    T0 = time()
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss, roc_auc_score
    
    ldf = []
    logs = {}

    weight_column = params["learner_params"]["extra_params"].get("weight_colum", None)
    linear_trees = params["learner_params"]["extra_params"].get("linear_trees", False)
    init_model= params["learner_params"]["extra_params"].get("init_model", None)
    init_score=params["learner_params"]["extra_params"].get("dataset_init_score", None)
    

    if linear_trees:
        logger.info(f"linear trees will be applied so training time may increase significantly.")


    skf = StratifiedKFold(n_splits = cv ,shuffle=True,random_state=random_state)
    train_df = train_df.reset_index(drop = True)
    logger.info(f"Starting pipeline: Generating {cv} k-fold training...")
    for i, (train_idx, validation_idx) in enumerate(skf.split(train_df.loc[:,features],
                                                  train_df.loc[:,target_column])):

        logger.info(f"Training for fold {i + 1}")
        tmp_train_df = train_df.loc[train_idx, :]
        tmp_validation = train_df.loc[validation_idx, :]

        tmp_bst, _ , model_logs = lgbm_classification_learner(tmp_train_df,
                                    features = features,
                                    target = target_column,
                                    linear_trees= linear_trees,
                                    weight_column =weight_column,
                                    init_model = init_model,
                                    dataset_init_score = init_score,
                                    learning_rate = params["learner_params"]["learning_rate"],
                                    num_estimators = params["learner_params"]["n_estimators"],
                                    extra_params = params["learner_params"]["extra_params"]
                                    ) 
        
        tmp_predictions = tmp_bst(tmp_validation, apply_shap = apply_shap)
        ldf.append(tmp_predictions)

    oof_preds_df = pd.concat(ldf, ignore_index = True)
    assert len(train_df) == len(oof_preds_df)

    logger.info("CV training finished!")
    logger.info("Training the model in the full dataset...")
    bst, _, model_logs = lgbm_classification_learner(
                                    train_df,
                                    features = features,
                                    target = target_column,
                                    linear_trees= linear_trees,
                                    weight_column =weight_column,
                                    learning_rate = params["learner_params"]["learning_rate"],
                                    num_estimators = params["learner_params"]["n_estimators"],
                                    extra_params = params["learner_params"]["extra_params"]
                                           )

    validation_preds_df = bst(validation_df, apply_shap = apply_shap)
    logger.info("Training process finished!")
    logger.info("Calculating metrics...")
    roc_auc_oof = get_roc(oof_preds_df[target_column], oof_preds_df["prediction"])
    roc_auc_val = get_roc(validation_preds_df[target_column], validation_preds_df["prediction"])

    log_loss_oof = get_logloss(oof_preds_df[target_column], oof_preds_df["prediction"])
    log_loss_val = get_logloss(validation_preds_df[target_column], validation_preds_df["prediction"])

    logs = {
		"lgbm_classification_learner": {
			"model_logs": model_logs,
			"predict_fn": bst
		},
		"data": {
			"oof_df": oof_preds_df,
			"validation_df": validation_preds_df
		},
		"metrics": {
			"roc_auc": {
				"out_of_fold": roc_auc_oof,
				"validation": roc_auc_val
			},
			"log_loss": {
				"out_of_fold": log_loss_oof,
				"validation": log_loss_val
			}
		}
    }

    logger.info(f"Full process finished in {(time() - T0) / 60:.2f} minutes.")

    logger.info(f"Saving the predict function.")
    if isinstance(save_estimator_path, str):
        with open(save_estimator_path, "wb") as context:
            cp.dump(bst, context)
    logger.info(f"Predict function saved.")
    
    return logs

def get_roc(y_true, y_pred):
    """
    """
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception as e:
        print(e)

def get_logloss(y_true, y_pred):
    """
    """
    from sklearn.metrics import log_loss
    try:
        return log_loss(y_true, y_pred)
    except:
        return np.NaN
        

    

    
    

