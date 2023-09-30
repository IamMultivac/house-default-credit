import numpy as np
import pandas as pd
import sys

from functools import reduce

from lightgbm import LGBMClassifier as lgbm

from time import time

sys.path.append("../")

# local imports
from preprocess.pos_cash.process import ProcessPosCash
from utils.functions__utils import get_linear_regression, train_binary
from utils.sk_id_curr_list import sk_id_curr_list
from utils.sk_id_curr_train_list import sk_id_curr_train_list
from utils.sk_id_curr_test_list import sk_id_curr_test_list
from src.learner_params import target_column, space_column, prediction_column, base_learners_params
from src.config import logger
from utils.features_lists import pos_base_features

new_columns = ['cnt_instalment','cnt_instalment_future', 'sk_dpd', 'sk_dpd_def', 'dpd_total']

agg_fns = ["sum","mean","min","max","std"]

merge_fn = lambda x,y:pd.merge(x,y, on = space_column, how = "inner")

bst = lgbm(**base_learners_params)

prefix = "pos_cash"

def main(pos_history_df:pd.DataFrame,         
         application_train_df:pd.DataFrame,
         application_test_df:pd.DataFrame,
         verbose:bool = True
        ):
    """
    """
    T0 = time()
    pos = ProcessPosCash(pos_history_df)
    pos_history = pos.process()
    logger.info("Training a base learner...")
    application_train_df.columns = application_train_df.columns.str.lower()
    application_test_df.columns = application_test_df.columns.str.lower()
    base_learner_logs = train_binary(merge_fn(pos_history, application_train_df[[space_column,target_column]]),
                                       pos_base_features,
                                       target_column,
                                       bst
                                      )
    base_learner_train= base_learner_logs["data"]
    
    base_learner_test = base_learner_logs["p"](merge_fn(pos_history, application_test_df[[space_column]]))

    base_learner_df = pd.concat([base_learner_train, base_learner_test], ignore_index = True)[[space_column, prediction_column]]


    frame = pd.DataFrame(index = sk_id_curr_list)
    logger.info("Creating features...")

    frame["cnt_installments_pos_mean"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment.sum().groupby("sk_id_curr").mean()
    frame["cnt_installments_pos_max"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment.sum().groupby("sk_id_curr").max()
    frame["cnt_installments_pos_total"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment.sum().groupby("sk_id_curr").sum()
    frame["cnt_installments_pos_trend"] = pos_history.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["cnt_installments_fut_pos_mean"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment_future.sum().groupby("sk_id_curr").mean()
    frame["cnt_installments_fut_pos_max"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment_future.sum().groupby("sk_id_curr").max()
    frame["cnt_installments_fut_pos_total"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment_future.sum().groupby("sk_id_curr").sum()
    frame["cnt_installments_fut_pos_trend"] = pos_history.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment_future.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    frame["cnt_installments_pos_minus_pos_f"] = frame["cnt_installments_pos_total"] - frame["cnt_installments_fut_pos_total"]
    
    frame["cnt_installments_dpd_pos_mean"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd.sum().groupby("sk_id_curr").mean()
    frame["cnt_installments_dpd_pos_max"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd.sum().groupby("sk_id_curr").max()
    frame["cnt_installments_dpd_pos_total"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd.sum().groupby("sk_id_curr").sum()
    frame["cnt_installments_dpd_pos_trend"] = pos_history.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["cnt_installments_par_pos_mean"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).par_x_at_y.sum().groupby("sk_id_curr").mean()
    frame["cnt_installments_par_pos_max"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).par_x_at_y.sum().groupby("sk_id_curr").max()
    frame["cnt_installments_par_pos_total"] = pos_history.groupby(["sk_id_curr", "sk_id_prev"]).par_x_at_y.sum().groupby("sk_id_curr").sum()
    frame["cnt_installments_par_pos_trend"] = pos_history.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).par_x_at_y.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()

    frame["mean_sk_dpd_contract_active"] =pos_history.groupby("sk_id_curr").apply(lambda x: x[x["name_contract_status"] == "Active"]["sk_dpd"].mean())
    frame["mean_sk_dpd_contract_completed"]=pos_history.groupby("sk_id_curr").apply(lambda x: x[x["name_contract_status"] == "Completed"]["sk_dpd"].mean())
    frame["mean_sk_dpd_contract_signed"]=pos_history.groupby("sk_id_curr").apply(lambda x: x[x["name_contract_status"] == "Signed"]["sk_dpd"].mean())
    frame["mean_sk_dpd_contract_demand"]=pos_history.groupby("sk_id_curr").apply(lambda x: x[x["name_contract_status"] == "Demand"]["sk_dpd"].mean())
    frame["mean_sk_dpd_contract_other"]=pos_history.groupby("sk_id_curr").apply(lambda x: x[~x["name_contract_status"].isin(["Active","Completed","Signed","Demand"])]["sk_dpd"].mean())
    
    frame["count_sk_dpd_contract_active"]=pos_history.groupby("sk_id_curr").apply(lambda x: x[x["name_contract_status"] == "Active"]["sk_dpd"].count())
    frame["count_sk_dpd_contract_completed"]=pos_history.groupby("sk_id_curr").apply(lambda x: x[x["name_contract_status"] == "Completed"]["sk_dpd"].count())
    frame["count_sk_dpd_contract_signed"]=pos_history.groupby("sk_id_curr").apply(lambda x: x[x["name_contract_status"] == "Signed"]["sk_dpd"].count())
    frame["count_sk_dpd_contract_demand"]=pos_history.groupby("sk_id_curr").apply(lambda x: x[x["name_contract_status"] == "Demand"]["sk_dpd"].count())
    frame["count_sk_dpd_contract_other"]=pos_history.groupby("sk_id_curr").apply(lambda x: x[~x["name_contract_status"].isin(["Active","Completed","Signed","Demand"])]["sk_dpd"].count())
    
    ldf = []
    for col in new_columns:
        for agg_fn in agg_fns:
            tmp = pos_history.groupby("sk_id_curr").agg({col:agg_fn}).rename(columns = {col:f"{col}_{agg_fn}"})
            ldf.append(tmp)
    
    agg_df = reduce(lambda x,y:pd.merge(x,y, left_index = True, right_index=True, how = "outer"), ldf)

    frame.replace(np.inf,np.NaN, inplace = True)
    frame.replace(-np.inf,np.NaN, inplace = True)
    
    frame = pd.merge(frame, agg_df, left_index = True, right_index = True, how = "outer")

    ldf = []
    for agg_fn in agg_fns:
        tmp = base_learner_df.groupby("sk_id_curr").agg({"prediction":agg_fn}).rename(columns = {prediction_column:f"{prediction_column}_{agg_fn}"})
        ldf.append(tmp)

    agg_df = reduce(lambda x,y:pd.merge(x,y, left_index = True, right_index=True, how = "outer"), ldf)

    frame = pd.merge(frame, agg_df, left_index = True, right_index = True, how = "outer")


    frame.columns = [f"{prefix}__{x}" for x in frame.columns.tolist()]

    frame = frame.reset_index().rename(columns = {"index":"sk_id_curr"})

    logger.info(f"Successfully created featureset of length: {len(frame)} in: {((time() - T0) / 60):.2f} minutes")

    if verbose:
        frame.info(verbose = True, show_counts = True)


    return frame

