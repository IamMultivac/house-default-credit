import pandas as pd
import numpy as np
import sys

from functools import reduce

from time import time

from lightgbm import LGBMClassifier as lgbm

sys.path.append("../")

# local imports
from preprocess.previous_application.process import ProcessPreviousApplication
from utils.functions__utils import get_linear_regression, train_binary
from utils.sk_id_curr_list import sk_id_curr_list
from utils.sk_id_curr_train_list import sk_id_curr_train_list
from utils.sk_id_curr_test_list import sk_id_curr_test_list
from src.learner_params import target_column, space_column, prediction_column, base_learners_params
from src.config import logger
from utils.features_lists import previous_application_base_features

prefix = "previous_application"

columns = ['days_first_drawing',
    'rate_down_payment',
    'rate_interest_privileged',
    'rate_interest_primary',
    'nflag_last_appl_in_day',
    'requested_vs_final_amt']


agg_fns = ["sum","mean","min","max","std"]
merge_fn = lambda x,y:pd.merge(x,y, on = space_column, how = "inner")

bst = lgbm(**base_learners_params)

def main(previous_application_df:pd.DataFrame,         
         application_train_df:pd.DataFrame,
         application_test_df:pd.DataFrame, 
         verbose:bool = True):
    """
    """
    T0 = time()
    pa = ProcessPreviousApplication(previous_application_df)
    previous_application = pa.process()

    logger.info("Training a base learner...")
    application_train_df.columns = application_train_df.columns.str.lower()
    application_test_df.columns = application_test_df.columns.str.lower()
    base_learner_logs = train_binary(merge_fn(previous_application, application_train_df[[space_column,target_column]]),
                                       previous_application_base_features,
                                       target_column,
                                       bst
                                      )
    base_learner_train= base_learner_logs["data"]
    
    base_learner_test = base_learner_logs["p"](merge_fn(previous_application, application_test_df[[space_column]]))

    base_learner_df = pd.concat([base_learner_train, base_learner_test], ignore_index = True)[[space_column, prediction_column]]


    frame = pd.DataFrame(index = sk_id_curr_list)
    logger.info("Creating features...")

    frame["sum_prev_applications"] = previous_application.groupby("sk_id_curr").sk_id_prev.size()
    frame["mean_amt_prev_applications"] = previous_application.groupby("sk_id_curr").amt_application.mean()
    frame["mean_amt_cred_prev_applications"] = previous_application.groupby("sk_id_curr").amt_credit.mean()
    frame["mean_amt_annuity_prev_applications"] = previous_application.groupby("sk_id_curr").amt_annuity.mean()
    frame["sum_amt_downpayment_prev_applications"] = previous_application.groupby("sk_id_curr").amt_down_payment.sum()
    frame["mean_amt_goodsprice_prev_applications"] = previous_application.groupby("sk_id_curr").amt_goods_price.mean()
    frame["mean_days_last_due_prev_applications"] = previous_application.groupby("sk_id_curr").days_last_due.mean()
    frame["mean_days_first_due_prev_applications"] = previous_application.groupby("sk_id_curr").days_first_due.mean()
    frame["mean_term_prev_applications"] = previous_application.groupby("sk_id_curr").cnt_payment.mean()
    frame["total_previous_revolving_credits"] = previous_application[previous_application["name_contract_type"] == "Revolving loans"].groupby("sk_id_curr").sk_id_prev.size()
    
    frame["percentage_prev_loans_with_insurance"] = previous_application_df.groupby("sk_id_curr").nflag_insured_on_approval.mean()
    
    frame["number_cards_previous_application"] = previous_application_df.groupby("sk_id_curr").apply(lambda x: (x["name_portfolio"] == "Cards").sum())
    frame["number_pos_previous_application"] = previous_application_df.groupby("sk_id_curr").apply(lambda x: (x["name_portfolio"] == "POS").sum())
    frame["number_cash_previous_application"] = previous_application_df.groupby("sk_id_curr").apply(lambda x: (x["name_portfolio"] == "Cash").sum())
    frame["number_cars_previous_application"] = previous_application_df.groupby("sk_id_curr").apply(lambda x: (x["name_portfolio"] == "Cars").sum())
    frame["number_others_previous_application"] = previous_application_df.groupby("sk_id_curr").apply(lambda x: (x["name_portfolio"] == "XNA").sum())
    
    apparel_list =["Mobile", "Consumer Electronics","Computers","Audio/Video","Furniture"]
    xna_list =["XNA"]
    
    frame["num_applications_goods_category_apparel"] = previous_application_df[previous_application_df["name_goods_category"].isin(apparel_list)].groupby("sk_id_curr").name_goods_category.count()
    frame["num_applications_goods_category_xna"] = previous_application_df[previous_application_df["name_goods_category"].isin(xna_list)].groupby("sk_id_curr").name_goods_category.count()
    frame["num_applications_goods_category_other"] = previous_application_df[~(previous_application_df["name_goods_category"].isin(apparel_list)) & ~(previous_application_df["name_goods_category"].isin(xna_list))].groupby("sk_id_curr").name_goods_category.count()
    
    frame["seller_number_economic_activities"] = previous_application_df.groupby("sk_id_curr").name_seller_industry.nunique()
    frame["name_product_type_unique"] = previous_application_df.groupby("sk_id_curr").name_product_type.nunique()
    frame["number_new_applications"] = previous_application_df.groupby("sk_id_curr").apply(lambda x: (x["name_client_type"] == "New").sum())
    frame["number_refresher_applications"] = previous_application_df.groupby("sk_id_curr").apply(lambda x: (x["name_client_type"] != "New").sum())
    
    frame["mean_amt_credit_previous_revolving_credits"] = previous_application[previous_application["name_contract_type"] == "Revolving loans"].groupby("sk_id_curr")["amt_credit"].mean()
    frame["mean_amt_downpayment_previous_revolving_credits"] = previous_application[previous_application["name_contract_type"] == "Revolving loans"].groupby("sk_id_curr")["amt_down_payment"].mean()
    
    frame["perc_prev_app_refused"] = previous_application[previous_application.name_contract_status == "Refused"].groupby("sk_id_curr").sk_id_prev.size() / (frame["sum_prev_applications"] + 1)
    frame["perc_prev_app_approved"] = previous_application[previous_application.name_contract_status == "Approved"].groupby("sk_id_curr").sk_id_prev.size() / (frame["sum_prev_applications"] + 1)
    frame["perc_prev_app_other"] = previous_application[previous_application.name_contract_status.isin(["Canceled", "Unused offer"])].groupby("sk_id_curr").sk_id_prev.size() / (frame["sum_prev_applications"] + 1)
    
    frame["mean_amt_prev_app_refused"] = previous_application[previous_application.name_contract_status == "Refused"].groupby("sk_id_curr").amt_application.mean()
    frame["max_amt_prev_app_refused"] = previous_application[previous_application.name_contract_status == "Approved"].groupby("sk_id_curr").amt_application.max()
    frame["min_amt_prev_app_refused"] = previous_application[previous_application.name_contract_status.isin(["Canceled", "Unused offer"])].groupby("sk_id_curr").amt_application.min()
    
    frame["channel_aq_last_refused_app"] = previous_application[previous_application.name_contract_status == "Refused"].sort_values(by=["sk_id_curr", "sk_id_prev"], ascending=[0, 1]).drop_duplicates(subset="sk_id_curr").groupby("sk_id_curr").channel_type.apply(lambda x: x.unique()[0])

    last_application_df = (previous_application
    .sort_values(['sk_id_curr', 'days_decision'])
    .drop_duplicates(subset = "sk_id_curr", keep = "last")
    .select_dtypes(exclude = "object")
    .drop("sk_id_prev", axis = 1)
    ).set_index("sk_id_curr")
    
    last_application_df.columns = [f"last_application_{x}" for x in last_application_df.columns]

    frame = pd.merge(frame, last_application_df, left_index = True, right_index = True, how = "outer")

    first_application_df = (previous_application
    .sort_values(['sk_id_curr', 'days_decision'])
    .drop_duplicates(subset = "sk_id_curr", keep = "first")
    .select_dtypes(exclude = "object")
    .drop("sk_id_prev", axis = 1)
    ).set_index("sk_id_curr")
    
    first_application_df.columns = [f"first_application_{x}" for x in first_application_df.columns]

    frame = pd.merge(frame, first_application_df, left_index = True, right_index = True, how = "outer")
    
    ldf = []
    for agg_fn in agg_fns:
        tmp = base_learner_df.groupby("sk_id_curr").agg({"prediction":agg_fn}).rename(columns = {prediction_column:f"{prediction_column}_{agg_fn}"})
        ldf.append(tmp)

    agg_df = reduce(lambda x,y:pd.merge(x,y, left_index = True, right_index=True, how = "outer"), ldf)

    frame = pd.merge(frame, agg_df, left_index = True, right_index = True, how = "outer")

        
    ldf = []
    for agg_fn in agg_fns:
        for col in columns:
            tmp = previous_application.groupby("sk_id_curr").agg({col:agg_fn}).rename(columns = {col:f"{col}_{agg_fn}"})
            ldf.append(tmp)
    agg_df = reduce(lambda x,y:pd.merge(x,y, left_index = True, right_index=True, how = "outer"), ldf)

    frame = pd.merge(frame, agg_df, left_index = True, right_index = True, how = "outer")

    frame.columns = [f"{prefix}__{x}" for x in frame.columns.tolist()]

    frame = frame.reset_index().rename(columns = {"index":"sk_id_curr"})

    # Create aggregations for the last n previous applications

    #######

    frame.replace(np.inf,np.NaN, inplace = True)
    frame.replace(-np.inf,np.NaN, inplace = True)

    logger.info(f"Successfully created featureset of length: {len(frame)} in: {((time() - T0) / 60):.2f} minutes")

    
    if verbose:
        frame.info(verbose = True, show_counts = True)

    return frame
