import numpy as np
import pandas as pd
import sys

from lightgbm import LGBMClassifier as lgbm

from functools import reduce

from time import time
import warnings; warnings.filterwarnings("ignore")

sys.path.append("../")

# local imports
from preprocess.bureau.process import ProcessBureau
from utils.functions__utils import get_linear_regression, train_binary
from utils.sk_id_curr_list import sk_id_curr_list
from utils.sk_id_curr_train_list import sk_id_curr_train_list
from utils.sk_id_curr_test_list import sk_id_curr_test_list
from src.config import logger
from src.learner_params import target_column, space_column, prediction_column, base_learners_params
from utils.features_lists import bureau_base_features

bst = lgbm(**base_learners_params)

prefix = "bureau"

premium_loans = ["Real estate loan",
                 "Car loan",
                 "Loan for purchase of shares (margin lending)"]

working_capital_loans = ["Loan for working capital replenishment",
                         "Loan for the purchase of equipment",
                         "Loan for business development"]

bank_credits = ["Microloan",
                "Consumer credit",
                "Interbank credit",
                "Consumer credit"]

credit_cards = ["Credit card"]

others = ["Unknown type of loan", "Another type of loan",
       "Cash loan (non-earmarked)",
       "Mobile operator loan", "Interbank credit",
       "Loan for purchase of shares (margin lending)"]

agg_fns = ["min","max","mean","std"]

merge_fn = lambda x,y:pd.merge(x,y, on = space_column, how = "inner")

def main(bureau_df:pd.DataFrame,
         bureau_balance_df:pd.DataFrame,
         application_train_df:pd.DataFrame,
         application_test_df:pd.DataFrame,
         verbose:bool = True
        ):
    """
    Perform feature engineering on bureau and bureau balance data.

    Parameters:
    -----------
    bureau_df : pd.DataFrame
        DataFrame containing bureau data.

    bureau_balance_df : pd.DataFrame
        DataFrame containing bureau balance data.

    verbose : bool, optional
        If True, print verbose log messages. Default is True.

    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features.

    """
    T0 = time()
    bureau = ProcessBureau(bureau_df, bureau_balance_df)
    bureau_df = bureau.process()
    logger.info("Training a base learner...")
    application_train_df.columns = application_train_df.columns.str.lower()
    application_test_df.columns = application_test_df.columns.str.lower()
    base_learner_logs = train_binary(merge_fn(bureau_df, application_train_df[[space_column,target_column]]),
                                       bureau_base_features,
                                       target_column,
                                       bst
                                      )
    base_learner_train= base_learner_logs["data"]
    
    base_learner_test = base_learner_logs["p"](merge_fn(bureau_df, application_test_df[[space_column]]))

    base_learner_df = pd.concat([base_learner_train, base_learner_test], ignore_index = True)[[space_column, prediction_column]]
    
    frame = pd.DataFrame(index=sk_id_curr_list)
    logger.info("Creating features...")
    
    # Total accounts
    frame["total_accounts"] = bureau_df.groupby("sk_id_curr").size()
    # Open and closed accounts
    frame["active_accounts"] = bureau_df.groupby("sk_id_curr").apply(lambda x: (x["credit_active"] == "Active").sum())
    frame["closed_accounts"] = bureau_df.groupby("sk_id_curr").apply(lambda x: (x["credit_active"] == "Closed").sum())
    
    frame["max_amt_overdue"] = bureau_df.groupby("sk_id_curr")["amt_credit_max_overdue"].max()
    frame["min_amt_overdue"] = bureau_df.groupby("sk_id_curr")["amt_credit_max_overdue"].min()
    frame["mean_amt_overdue"] = bureau_df.groupby("sk_id_curr")["amt_credit_max_overdue"].mean()
    frame["std_amt_overdue"] = bureau_df.groupby("sk_id_curr")["amt_credit_max_overdue"].std()
    
    frame["mean_amt_credit_debt"] = bureau_df.groupby("sk_id_curr")["amt_credit_sum_debt"].mean()
    frame["max_amt_credit_debt"] = bureau_df.groupby("sk_id_curr")["amt_credit_sum_debt"].max()
    
    frame["sum_amt_credit_sum"] = bureau_df.groupby("sk_id_curr")["amt_credit_sum"].sum()
    frame["mean_amt_credit_sum"] = bureau_df.groupby("sk_id_curr")["amt_credit_sum"].mean()
    frame["max_amt_credit_min"] = bureau_df.groupby("sk_id_curr")["amt_credit_sum"].min()
    frame["max_amt_credit_sum"] = bureau_df.groupby("sk_id_curr")["amt_credit_sum"].max()
    frame["max_amt_credit_std"] = bureau_df.groupby("sk_id_curr")["amt_credit_sum"].std()
    
    
    frame["sum_amt_credit_debt"] = bureau_df.groupby("sk_id_curr")["amt_credit_sum_debt"].sum()
    frame["mean_amt_credit_limit"] = bureau_df.groupby("sk_id_curr")["amt_credit_sum_limit"].mean()
    frame["sum_amt_credit_limit"] = bureau_df.groupby("sk_id_curr")["amt_credit_sum_limit"].sum()
    
    frame["sum_amt_credit_overdue"] = bureau_df.groupby("sk_id_curr")["amt_credit_max_overdue"].sum()
    frame["mean_amt_credit_overdue"] = bureau_df.groupby("sk_id_curr")["amt_credit_max_overdue"].mean()
    
    frame["max_amt_credit_overdue"] = bureau_df.groupby("sk_id_curr")["amt_credit_max_overdue"].max()
    
    frame["mean_days_credit_overdue"] =bureau_df.groupby("sk_id_curr").credit_day_overdue.mean()
    frame["sum_days_credit_overdue"] =bureau_df.groupby("sk_id_curr").credit_day_overdue.sum()
    frame["max_days_credit_overdue"] =bureau_df.groupby("sk_id_curr").credit_day_overdue.max()
    
    
    frame["credit_sum_by_debt"] = frame["sum_amt_credit_debt"] / (frame["sum_amt_credit_sum"] + 1)
    frame["credit_sum_by_limit"] = frame["sum_amt_credit_limit"] / frame["sum_amt_credit_sum"]
    frame["credit_sum_by_overdue"] = frame["sum_amt_credit_overdue"] / frame["sum_amt_credit_sum"]
    
    # Divide by AMT_CREDIT_SUM
    frame["total_annuity"] = bureau_df.groupby("sk_id_curr")["amt_annuity"].sum()
    # Percentage open accounts
    frame["active_prem_credits"] = bureau_df[(bureau_df["credit_type"].isin(premium_loans)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].size() / frame["active_accounts"]
    frame["active_wk_credits"] = bureau_df[(bureau_df["credit_type"].isin(working_capital_loans)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].size() / frame["active_accounts"]
    frame["active_bank_credits"] = bureau_df[(bureau_df["credit_type"].isin(bank_credits)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].size() / frame["active_accounts"]
    frame["active_credit_cards"] = bureau_df[(bureau_df["credit_type"].isin(credit_cards)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].size() / frame["active_accounts"]
    frame["active_other_credits"] = bureau_df[(bureau_df["credit_type"].isin(others)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].size() / frame["active_accounts"]
    # Percentage closed accounts
    frame["closed_prem_credits"] = bureau_df[(bureau_df["credit_type"].isin(premium_loans)) & (bureau_df["credit_active"] == "Closed")].groupby("sk_id_curr")["amt_credit_sum_debt"].size() / frame["closed_accounts"]
    frame["closed_wk_credits"] = bureau_df[(bureau_df["credit_type"].isin(working_capital_loans)) & (bureau_df["credit_active"] == "Closed")].groupby("sk_id_curr")["amt_credit_sum_debt"].size() / frame["closed_accounts"]
    frame["closed_bank_credits"] = bureau_df[(bureau_df["credit_type"].isin(bank_credits)) & (bureau_df["credit_active"] == "Closed")].groupby("sk_id_curr")["amt_credit_sum_debt"].size() / frame["closed_accounts"]
    frame["closed_credit_cards"] = bureau_df[(bureau_df["credit_type"].isin(credit_cards)) & (bureau_df["credit_active"] == "Closed")].groupby("sk_id_curr")["amt_credit_sum_debt"].size() / frame["closed_accounts"]
    frame["closed_other_credits"] = bureau_df[(bureau_df["credit_type"].isin(others)) & (bureau_df["credit_active"] == "Closed")].groupby("sk_id_curr")["amt_credit_sum_debt"].size() / frame["closed_accounts"]
    
    # Saldo en $$ en cuentas abiertas
    frame["mean_debt_active_prem_credits"] = bureau_df[(bureau_df["credit_type"].isin(premium_loans)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].mean()
    frame["mean_debt_active_wk_credits"] = bureau_df[(bureau_df["credit_type"].isin(working_capital_loans)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].mean()
    frame["mean_debt_active_bank_credits"] = bureau_df[(bureau_df["credit_type"].isin(bank_credits)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].mean()
    frame["mean_debt_active_credit_cards"] = bureau_df[(bureau_df["credit_type"].isin(credit_cards)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].mean()
    frame["mean_debt_active_other_credits"] = bureau_df[(bureau_df["credit_type"].isin(others)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].mean()
    
    frame["sum_debt_active_prem_credits"] = bureau_df[(bureau_df["credit_type"].isin(premium_loans)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].sum()
    frame["sum_debt_active_wk_credits"] = bureau_df[(bureau_df["credit_type"].isin(working_capital_loans)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].sum()
    frame["sum_debt_active_bank_credits"] = bureau_df[(bureau_df["credit_type"].isin(bank_credits)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].sum()
    frame["sum_debt_active_credit_cards"] = bureau_df[(bureau_df["credit_type"].isin(credit_cards)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].sum()
    frame["sum_debt_active_other_credits"] = bureau_df[(bureau_df["credit_type"].isin(others)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].sum()
    
    frame["sum_debt_active_credits"] = bureau_df[(bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_debt"].sum()
    # Trend Saldo en $$ en cuentas abiertas
    frame["trend_debt_active_credits"] = bureau_df[(bureau_df["credit_active"] == "Active")].sort_values(by=["sk_id_curr", "days_credit"]).groupby("sk_id_curr")["amt_credit_sum_debt"].fillna(-1).apply(lambda x: get_linear_regression(x)["trend"])
    # Max en $$ en cuentas cerradas
    frame["max_debt_closed_premium_loans"] = bureau_df[(bureau_df["credit_type"].isin(premium_loans)) & (bureau_df["credit_active"] == "Closed")].groupby("sk_id_curr")["amt_credit_sum_debt"].max()
    frame["max_debt_closed_wk_credits"] = bureau_df[(bureau_df["credit_type"].isin(working_capital_loans)) & (bureau_df["credit_active"] == "Closed")].groupby("sk_id_curr")["amt_credit_sum_debt"].max()
    frame["max_debt_closed_bank_credits"] = bureau_df[(bureau_df["credit_type"].isin(bank_credits)) & (bureau_df["credit_active"] == "Closed")].groupby("sk_id_curr")["amt_credit_sum_debt"].max()
    frame["max_debt_closed_credit_card"] = bureau_df[(bureau_df["credit_type"].isin(credit_cards)) & (bureau_df["credit_active"] == "Closed")].groupby("sk_id_curr")["amt_credit_sum_debt"].max()
    frame["max_debt_closed_debt_others"] = bureau_df[(bureau_df["credit_type"].isin(others)) & (bureau_df["credit_active"] == "Closed")].groupby("sk_id_curr")["amt_credit_sum_debt"].max()
    # div
    frame["sum_amt_credit_limit_credit_cards_open"] = bureau_df[(bureau_df["credit_type"].isin(credit_cards)) & (bureau_df["credit_active"] == "Active")].groupby("sk_id_curr")["amt_credit_sum_limit"].sum()
    frame["perc_util_revolving"] = frame["sum_debt_active_credit_cards"] / frame["sum_amt_credit_limit_credit_cards_open"]
    frame["credit_debt_div_lim_cred"] = frame["sum_amt_credit_debt"] / frame["sum_amt_credit_limit"]
    frame["credit_lim_div_cred_overdue"] = frame["sum_amt_credit_limit"] / frame["sum_amt_credit_overdue"]
    
    # data.drop(["sum_amt_credit_sum", "sum_amt_credit_limit", "sum_amt_credit_limit_credit_cards_open"], axis=1, inplace=True)
    # Max cred lim non defaulted
    frame["max_cred_lim_non_delinq"] = bureau_df.groupby("sk_id_curr")[["ever_delinquent", "amt_credit_sum_limit"]].apply(lambda x: x[(x["ever_delinquent"] == 0)]["amt_credit_sum_limit"].max())
    frame["max_cred_lim_non_overdue"] = bureau_df.groupby("sk_id_curr")[["ever_delinquent", "amt_credit_max_overdue"]].apply(lambda x: x[(x["ever_delinquent"] == 0)]["amt_credit_max_overdue"].max())
    
    frame["max_cred_lim_delinq"] = bureau_df.groupby("sk_id_curr")[["ever_delinquent", "amt_credit_sum_limit"]].apply(lambda x: x[(x["ever_delinquent"] == 1)]["amt_credit_sum_limit"].max())
    frame["max_cred_lim_overdue"] = bureau_df.groupby("sk_id_curr")[["ever_delinquent", "amt_credit_max_overdue"]].apply(lambda x: x[(x["ever_delinquent"] == 1)]["amt_credit_max_overdue"].max())
    
    frame["mean_cred_lim_delinq"] = bureau_df.groupby("sk_id_curr")[["ever_delinquent", "amt_credit_sum_limit"]].apply(lambda x: x[(x["ever_delinquent"] == 1)]["amt_credit_sum_limit"].mean())
    frame["mean_cred_lim_overdue"] = bureau_df.groupby("sk_id_curr")[["ever_delinquent", "amt_credit_max_overdue"]].apply(lambda x: x[(x["ever_delinquent"] == 1)]["amt_credit_max_overdue"].mean())
    
    # Payment history
    frame["payment_history_mean"] = bureau_df.groupby("sk_id_curr")["payment_history"].mean()
    frame["payment_history_std"] = bureau_df.groupby("sk_id_curr")["payment_history"].std()
    frame["mean_consecutive_non_delinq"] = bureau_df.groupby("sk_id_curr")["consecutive_no_delinq"].mean()
    frame["max_consecutive_non_delinq"] = bureau_df.groupby("sk_id_curr")["consecutive_no_delinq"].max()
    
    # Number of times in bucket x
    frame["sum_times_bucket_1"] = bureau_df.groupby("sk_id_curr")["times_bucket_1"].sum()
    frame["sum_times_bucket_2"] = bureau_df.groupby("sk_id_curr")["times_bucket_2"].sum()
    frame["sum_times_bucket_3"] = bureau_df.groupby("sk_id_curr")["times_bucket_3"].sum()
    frame["sum_times_bucket_4"] = bureau_df.groupby("sk_id_curr")["times_bucket_4"].sum()
    frame["sum_times_bucket_5"] = bureau_df.groupby("sk_id_curr")["times_bucket_5"].sum()
    frame["sum_times_bucket_6"] = bureau_df.groupby("sk_id_curr")["times_bucket_6"].sum()
    
    frame["times_bad_delinquency"] = frame["sum_times_bucket_6"] + frame["sum_times_bucket_5"] + frame["sum_times_bucket_4"] + frame["sum_times_bucket_3"]
    frame["times_no_bad_delinquency"] = frame["sum_times_bucket_2"] + frame["sum_times_bucket_1"]
    frame["total_times_delinquency"] = frame["sum_times_bucket_6"] + frame["sum_times_bucket_5"] + frame["sum_times_bucket_4"] + frame["sum_times_bucket_3"]+ frame["sum_times_bucket_2"]+ frame["sum_times_bucket_2"]
    # Accounts tenure
    frame["antiguedad_media_dias"] = bureau_df.groupby("sk_id_curr")["days_credit"].mean()
    frame["antiguedad_media_dias_closed_accounts"] = bureau_df.groupby("sk_id_curr")["days_credit_enddate"].mean()
    frame["antiguedad_media_dias_open_accounts"] = bureau_df.groupby("sk_id_curr")["days_enddate_fact"].mean()
    frame["antiguedad_maxima_dias_closed_accounts"] = bureau_df.groupby("sk_id_curr")["days_credit_enddate"].max()
    frame["antiguedad_maxima_dias_open_accounts"] = bureau_df.groupby("sk_id_curr")["days_enddate_fact"].max()
    frame["sum_prolong_days"] = bureau_df.groupby("sk_id_curr")["cnt_credit_prolong"].sum()
    frame["mean_prolong_days"] = bureau_df.groupby("sk_id_curr")["cnt_credit_prolong"].mean()
    # Closed accounts last x days
    frame["closed_accounts_last12m"] = bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_enddate_fact"] >= -365).sum())
    frame["closed_accounts_last9m"] =  bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_enddate_fact"] >= -270).sum())
    frame["closed_accounts_last6m"] =  bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_enddate_fact"] >= -180).sum())
    frame["closed_accounts_last3m"] =  bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_enddate_fact"] >= -90).sum())
    frame["closed_accounts_last2m"] =  bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_enddate_fact"] >= -60).sum())
    frame["closed_accounts_last1m"] =  bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_enddate_fact"] >= -30).sum())
    # Open credit last x days
    frame["open_accounts_last12m"] = bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_credit"] >= -365).sum())
    frame["open_accounts_last9m"] = bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_credit"] >= -270).sum())
    frame["open_accounts_last6m"] = bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_credit"] >= -180).sum())
    frame["open_accounts_last3m"] = bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_credit"] >= -90).sum())
    frame["open_accounts_last2m"] = bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_credit"] >= -60).sum())
    frame["open_accounts_last1m"] = bureau_df.groupby("sk_id_curr").apply(lambda x:(x["days_credit"] >= -30).sum())
    frame["n_unique_credit_types"] = bureau_df.groupby("sk_id_curr")["credit_type"].nunique()

    # hhi index
    frame["hhi_amt_credit_sum"] =bureau_df.groupby("sk_id_curr").apply(lambda x: ((x["amt_credit_sum"]/x["amt_credit_sum"].sum())**2).sum())
    frame["hhi_amt_credit_sum_debt"] =bureau_df.groupby("sk_id_curr").apply(lambda x: ((x["amt_credit_sum_debt"]/x["amt_credit_sum_debt"].sum())**2).sum())
    frame["hhi_amt_credit_sum_limit"] =bureau_df.groupby("sk_id_curr").apply(lambda x: ((x["amt_credit_sum_limit"]/x["amt_credit_sum_limit"].sum())**2).sum())
    frame["hhi_amt_credit_sum_overdue"] =bureau_df.groupby("sk_id_curr").apply(lambda x: ((x["amt_credit_sum_overdue"]/x["amt_credit_sum_overdue"].sum())**2).sum())
    # concentration
    frame["top_1_concentration_credit_sum"] =bureau_df.groupby("sk_id_curr").apply(lambda x: x["amt_credit_sum"].max()/x["amt_credit_sum"].sum())
    frame["top_1_concentration_credit_sum_debt"] =bureau_df.groupby("sk_id_curr").apply(lambda x: x["amt_credit_sum_debt"].max()/x["amt_credit_sum_debt"].sum())
    frame["top_1_concentration_credit_sum_limit"] =bureau_df.groupby("sk_id_curr").apply(lambda x: x["amt_credit_sum_limit"].max()/x["amt_credit_sum_limit"].sum())
    frame["top_1_concentration_credit_sum_overdue"] =bureau_df.groupby("sk_id_curr").apply(lambda x: x["amt_credit_sum_overdue"].max()/x["amt_credit_sum_overdue"].sum())

    ldf = []
    for agg_fn in agg_fns:
        tmp = base_learner_df.groupby("sk_id_curr").agg({"prediction":agg_fn}).rename(columns = {prediction_column:f"{prediction_column}_{agg_fn}"})
        ldf.append(tmp)

    agg_df = reduce(lambda x,y:pd.merge(x,y, left_index = True, right_index=True, how = "outer"), ldf)

    frame = pd.merge(frame, agg_df, left_index = True, right_index = True, how = "outer")

    frame.replace(np.inf,np.NaN, inplace = True)
    frame.replace(-np.inf,np.NaN, inplace = True)

    frame.columns = [f"{prefix}__{x}" for x in frame.columns.tolist()]

    frame = frame.reset_index().rename(columns = {"index":"sk_id_curr"})

    logger.info(f"Successfully created featureset of length: {len(frame)} in: {((time() - T0) / 60):.2f} minutes")

    if verbose:
        frame.info(verbose = True, show_counts = True)

    return frame

