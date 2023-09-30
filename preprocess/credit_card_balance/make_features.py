import numpy as np
import pandas as pd
import sys
from time import time

from lightgbm import LGBMClassifier as lgbm

from functools import reduce

sys.path.append("../")

# local imports
from preprocess.credit_card_balance.process import ProcessCreditCardBalance
from utils.functions__utils import get_linear_regression, train_binary
from utils.sk_id_curr_list import sk_id_curr_list
from utils.sk_id_curr_train_list import sk_id_curr_train_list
from utils.sk_id_curr_test_list import sk_id_curr_test_list
from src.learner_params import target_column, space_column, prediction_column, base_learners_params
from utils.features_lists import credit_card_base_features
from src.config import logger

prefix = "credit_card"

new_columns = ['amt_balance',
                'limit_use',
                'amt_credit_limit_actual',
                'amt_drawings_atm_current',
                'amt_drawings_current',
                'amt_drawings_pos_current',
                'amt_inst_min_regularity',
                'amt_payment_total_current',
                'amt_total_receivable',
                'cnt_drawings_atm_current',
                'cnt_drawings_current',
                'cnt_drawings_pos_current',
                'sk_dpd',
                'drawing_limit_ratio',
                'late_payment',
                'card_is_dpd_under_120',
                'card_is_dpd_over_120']
    
agg_fns = ["min","max","mean","std"]

merge_fn = lambda x,y:pd.merge(x,y, on = space_column, how = "inner")

bst = lgbm(**base_learners_params)


def main(credit_card_balance_df:pd.DataFrame,
         application_train_df:pd.DataFrame,
         application_test_df:pd.DataFrame,
         verbose:bool = True
        ):
    """
    """
    T0 = time()
    ccb = ProcessCreditCardBalance(credit_card_balance_df)
    credit_card_balance = ccb.process()
    logger.info("Training a base learner...")
    application_train_df.columns = application_train_df.columns.str.lower()
    application_test_df.columns = application_test_df.columns.str.lower()
    base_learner_logs = train_binary(merge_fn(credit_card_balance_df, application_train_df[[space_column,target_column]]),
                                       credit_card_base_features,
                                       target_column,
                                       bst
                                      )
    base_learner_train= base_learner_logs["data"]
    
    base_learner_test = base_learner_logs["p"](merge_fn(credit_card_balance_df, application_test_df[[space_column]]))

    base_learner_df = pd.concat([base_learner_train, base_learner_test], ignore_index = True)[[space_column, prediction_column]]


    frame = pd.DataFrame(index = sk_id_curr_list)
    logger.info("Creating features...")

    
    frame["mean_credit_cards_months"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_balance.size().groupby("sk_id_curr").mean()
    frame["max_credit_cards_months"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_balance.size().groupby("sk_id_curr").max()
    frame["total_credit_cards_amt_balance"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_balance.sum().groupby("sk_id_curr").sum()
    frame["max_credit_cards_amt_balance"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_balance.sum().groupby("sk_id_curr").max()
    frame["mean_credit_cards_amt_balance"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_balance.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_balance"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_balance.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_amt_balance"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_credit_limit_actual.sum().groupby("sk_id_curr").max()
    frame["mean_credit_cards_amt_balance"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_credit_limit_actual.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_balance"] = credit_card_balance.sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_credit_limit_actual.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    frame["max_credit_cards_amt_paymentcurrent"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_current.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_amt_paymentcurrent"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_current.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_amt_paymentcurrent"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_paymentcurrent"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_amt_paymentcurrent"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_current.sum().groupby("sk_id_curr").max()
    frame["mean_credit_cards_amt_paymentcurrent"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_paymentcurrent"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_amt_total_recievable"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_total_receivable.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_amt_total_recievable"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_total_receivable.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_amt_total_recievable"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_total_receivable.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_total_recievable"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_total_receivable.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_amt_min_regularity"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_inst_min_regularity.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_amt_min_regularity"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_inst_min_regularity.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_amt_min_regularity"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_inst_min_regularity.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_min_regularity"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_inst_min_regularity.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_amt_payment_total_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_total_current.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_amt_payment_total_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_total_current.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_amt_payment_total_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_total_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_payment_total_current"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_payment_total_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_amt_drawings_atm_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_atm_current.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_amt_drawings_atm_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_atm_current.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_amt_drawings_atm_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_atm_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_drawings_atm_current"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_atm_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_amt_drawings_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_current.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_amt_drawings_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_current.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_amt_drawings_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_drawings_current"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_amt_drawings_other_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_other_current.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_amt_drawings_other_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_other_current.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_amt_drawings_other_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_other_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_drawings_other_current"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_other_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_amt_drawings_pos_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_pos_current.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_amt_drawings_pos_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_pos_current.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_amt_drawings_pos_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_pos_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_drawings_pos_total_current"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_drawings_pos_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_amt_recivable"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_recivable.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_amt_recivable"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_recivable.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_amt_recivable"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).amt_recivable.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_amt_recivable"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).amt_recivable.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_cnt_atm_drawings_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_atm_current.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_cnt_atm_drawings_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_atm_current.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_cnt_atm_drawings_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_atm_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_cnt_atm_drawings_current"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_atm_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_cnt_drawings_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_current.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_cnt_drawings_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_current.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_cnt_drawings_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_cnt_drawings_current"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_cnt_drawings_other_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_other_current.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_cnt_drawings_other_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_other_current.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_cnt_drawings_other_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_other_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_cnt_drawings_other_current"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_other_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_cnt_drawings_pos_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_pos_current.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_cnt_drawings_pos_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_pos_current.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_cnt_drawings_pos_current"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_pos_current.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_cnt_drawings_pos_current"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).cnt_drawings_pos_current.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_cnt_installment_mature_cum"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment_mature_cum.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_cnt_installment_mature_cum"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment_mature_cum.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_cnt_installment_mature_cum"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment_mature_cum.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_cards_cnt_installment_mature_cum"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).cnt_instalment_mature_cum.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_dpd_def"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd_def.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_dpd_def"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd_def.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_dpd_def"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd_def.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_dpd_def"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd_def.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["max_credit_cards_dpd"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd.sum().groupby("sk_id_curr").max()
    frame["total_credit_cards_dpd"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd.sum().groupby("sk_id_curr").sum()
    frame["mean_credit_cards_dpd"] = credit_card_balance.groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd.sum().groupby("sk_id_curr").mean()
    frame["mean_trend_credit_dpd"] = credit_card_balance.fillna(0).sort_values(by=["sk_id_curr", "sk_id_prev"]).groupby(["sk_id_curr", "sk_id_prev"]).sk_dpd.apply(lambda x: get_linear_regression(x)["trend"]).groupby("sk_id_curr").mean()
    
    frame["amt_paycurr_div_total_amt_balance"] = frame["max_credit_cards_amt_paymentcurrent"]/frame["total_credit_cards_amt_balance"]
    frame["amt_total_balnce_div_total_recivable"] = frame["total_credit_cards_amt_balance"]/frame["total_credit_cards_amt_total_recievable"]


    ldf = []
    for col in new_columns:
        for agg_fn in agg_fns:
            tmp = credit_card_balance.groupby("sk_id_curr").agg({col:agg_fn}).rename(columns = {col:f"{col}_{agg_fn}"})
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
