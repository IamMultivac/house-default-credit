import pandas as pd
import numpy as np
import sys

from functools import reduce

from time import time

from lightgbm import LGBMClassifier as lgbm

sys.path.append("../")

# local imports
from preprocess.installments.process import ProcessInstallments
from utils.functions__utils import get_linear_regression, train_binary
from utils.sk_id_curr_list import sk_id_curr_list
from utils.sk_id_curr_train_list import sk_id_curr_train_list
from utils.sk_id_curr_test_list import sk_id_curr_test_list
from src.learner_params import target_column, space_column, prediction_column, base_learners_params
from src.config import logger
from utils.features_lists import installments_base_features

prefix = "installments"

new_columns = ['paid_over_amount',
'payment_difference_credit',
'payment_ratio_credit',
'paid_over',
'payment_perc',
'payment_diff',
'dpd_diff',
'dbd_diff',
'dpd',
'dbd',
'late_payment',
'instalment_payment_ratio',
'late_payment_ratio',
'significant_late_payment']

agg_fns = ["sum","max","mean","std"]

merge_fn = lambda x,y:pd.merge(x,y, on = space_column, how = "inner")

bst = lgbm(**base_learners_params)

def main(installments_payments_df:pd.DataFrame,
         main_application_df:pd.DataFrame,
         verbose:bool = True
        ):
    """
    """
    T0 = time()
    installments = ProcessInstallments(installments_payments_df, main_application_df)
    installments_payments = installments.process()
    logger.info("Training a base learner...")
    application_train_df = main_application_df[main_application_df[space_column].isin(sk_id_curr_train_list)]
    application_test_df = main_application_df[main_application_df[space_column].isin(sk_id_curr_test_list)]
    application_train_df.columns = application_train_df.columns.str.lower()
    application_test_df.columns = application_test_df.columns.str.lower()
    base_learner_logs = train_binary(merge_fn(installments_payments, application_train_df[[space_column,target_column]]),
                                       installments_base_features,
                                       target_column,
                                       bst
                                      )
    base_learner_train= base_learner_logs["data"]
    
    base_learner_test = base_learner_logs["p"](merge_fn(installments_payments, application_test_df[[space_column]]))

    base_learner_df = pd.concat([base_learner_train, base_learner_test], ignore_index = True)[[space_column, prediction_column]]

    frame = pd.DataFrame(index = sk_id_curr_list)
    logger.info("Creating features...")
    
    # installmet features
    frame["number_installments"] = installments_payments.groupby("sk_id_curr").sk_id_prev.count()
    frame["amt_installments_max_amt"] = installments_payments.groupby(["sk_id_curr", "sk_id_prev"]).amt_instalment.sum().groupby("sk_id_curr").max()
    frame["amt_installments_min_amt"] = installments_payments.groupby(["sk_id_curr", "sk_id_prev"]).amt_instalment.sum().groupby("sk_id_curr").min()
    frame["amt_installments_total_amt"] = installments_payments.groupby(["sk_id_curr", "sk_id_prev"]).amt_instalment.sum().groupby("sk_id_curr").sum()
    frame["amt_installments_mean_amt"] = installments_payments.groupby(["sk_id_curr", "sk_id_prev"]).amt_instalment.sum().groupby("sk_id_curr").mean()
    frame["amt_installments_trend"] = installments_payments.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev", "num_instalment_number"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_instalment.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["amt_pay_installments_max_amt"] = installments_payments.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment.sum().groupby("sk_id_curr").max()
    frame["amt_pay_installments_min_amt"] = installments_payments.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment.sum().groupby("sk_id_curr").min()
    frame["amt_pay_installments_total_amt"] = installments_payments.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment.sum().groupby("sk_id_curr").sum()
    frame["amt_pay_installments_mean_amt"] = installments_payments.groupby(["sk_id_curr", "sk_id_prev"]).amt_instalment.sum().groupby("sk_id_curr").mean()
    frame["amt_pay_installments_trend"] = installments_payments.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev", "num_instalment_number"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_payment.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()

    # hhi index
    frame["hhi_amt_amt_payment"] =installments_payments.groupby("sk_id_curr").apply(lambda x: ((x["amt_payment"]/x["amt_payment"].sum())**2).sum())
    frame["hhi_amt_instalment"] =installments_payments.groupby("sk_id_curr").apply(lambda x: ((x["amt_instalment"]/x["amt_instalment"].sum())**2).sum())
    
    frame["installments_payment_vs_amt_installment"] = frame["amt_pay_installments_total_amt"] / frame["amt_installments_total_amt"]

    ldf = []
    for col in new_columns:
        for agg_fn in agg_fns:
            tmp = installments_payments.groupby("sk_id_curr").agg({col:agg_fn}).rename(columns = {col:f"{col}_{agg_fn}"})
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
