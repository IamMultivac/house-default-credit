# %% [code]
# %% [code] {"_kg_hide-input":false}
import numpy as np
import pandas as pd

from functools import reduce
from itertools import groupby

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from scipy.stats import entropy
import pickle
import gzip

import pdb

from time import time




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




flag_doc_feat = [
 'FLAG_DOCUMENT_2',
 'FLAG_DOCUMENT_3',
 'FLAG_DOCUMENT_4',
 'FLAG_DOCUMENT_5',
 'FLAG_DOCUMENT_6',
 'FLAG_DOCUMENT_7',
 'FLAG_DOCUMENT_8',
 'FLAG_DOCUMENT_9',
 'FLAG_DOCUMENT_10',
 'FLAG_DOCUMENT_11',
 'FLAG_DOCUMENT_12',
 'FLAG_DOCUMENT_13',
 'FLAG_DOCUMENT_14',
 'FLAG_DOCUMENT_15',
 'FLAG_DOCUMENT_16',
 'FLAG_DOCUMENT_17',
 'FLAG_DOCUMENT_18',
 'FLAG_DOCUMENT_19',
 'FLAG_DOCUMENT_20',
 'FLAG_DOCUMENT_21',
 'REG_REGION_NOT_LIVE_REGION',
 'REG_REGION_NOT_WORK_REGION',
 'LIVE_REGION_NOT_WORK_REGION',
 'REG_CITY_NOT_LIVE_CITY',
 'REG_CITY_NOT_WORK_CITY',
 'LIVE_CITY_NOT_WORK_CITY',
 'FLAG_MOBIL',
 'FLAG_EMP_PHONE',
 'FLAG_WORK_PHONE',
 'FLAG_CONT_MOBILE',
 'FLAG_PHONE',
 'FLAG_EMAIL']

social_demo_feat = [
 'APARTMENTS_AVG',
 'BASEMENTAREA_AVG',
 'YEARS_BEGINEXPLUATATION_AVG',
 'YEARS_BUILD_AVG',
 'COMMONAREA_AVG',
 'ELEVATORS_AVG',
 'ENTRANCES_AVG',
 'FLOORSMAX_AVG',
 'FLOORSMIN_AVG',
 'LANDAREA_AVG',
 'LIVINGAPARTMENTS_AVG',
 'LIVINGAREA_AVG',
 'NONLIVINGAPARTMENTS_AVG',
 'NONLIVINGAREA_AVG',
 'APARTMENTS_MODE',
 'BASEMENTAREA_MODE',
 'YEARS_BEGINEXPLUATATION_MODE',
 'YEARS_BUILD_MODE',
 'COMMONAREA_MODE',
 'ELEVATORS_MODE',
 'ENTRANCES_MODE',
 'FLOORSMAX_MODE',
 'FLOORSMIN_MODE',
 'LANDAREA_MODE',
 'LIVINGAPARTMENTS_MODE',
 'LIVINGAREA_MODE',
 'NONLIVINGAPARTMENTS_MODE',
 'NONLIVINGAREA_MODE',
 'APARTMENTS_MEDI',
 'BASEMENTAREA_MEDI',
 'YEARS_BEGINEXPLUATATION_MEDI',
 'YEARS_BUILD_MEDI',
 'COMMONAREA_MEDI',
 'ELEVATORS_MEDI',
 'ENTRANCES_MEDI',
 'FLOORSMAX_MEDI',
 'FLOORSMIN_MEDI',
 'LANDAREA_MEDI',
 'LIVINGAPARTMENTS_MEDI',
 'LIVINGAREA_MEDI',
 'NONLIVINGAPARTMENTS_MEDI',
 'NONLIVINGAREA_MEDI',
 'TOTALAREA_MODE',
 'OBS_30_CNT_SOCIAL_CIRCLE',
 'DEF_30_CNT_SOCIAL_CIRCLE',
 'OBS_60_CNT_SOCIAL_CIRCLE',
 'DEF_60_CNT_SOCIAL_CIRCLE',
 'REGION_POPULATION_RELATIVE',
 'CNT_CHILDREN',
 'CNT_FAM_MEMBERS',
 'REGION_RATING_CLIENT',
 'REGION_RATING_CLIENT_W_CITY',
 'HOUR_APPR_PROCESS_START',
 'DAYS_LAST_PHONE_CHANGE',
 'CODE_GENDER',
 'FLAG_OWN_CAR',
 'FLAG_OWN_REALTY']

main_application_feat = ['SK_ID_CURR','AMT_REQ_CREDIT_BUREAU_HOUR',
 'AMT_REQ_CREDIT_BUREAU_DAY',
 'AMT_REQ_CREDIT_BUREAU_WEEK',
 'AMT_REQ_CREDIT_BUREAU_MON',
 'AMT_REQ_CREDIT_BUREAU_QRT',
 'AMT_REQ_CREDIT_BUREAU_YEAR',
 'AMT_INCOME_TOTAL',
 'AMT_CREDIT',
 'AMT_ANNUITY',
 'AMT_GOODS_PRICE',
 'DAYS_BIRTH',
 'DAYS_EMPLOYED',
 'DAYS_REGISTRATION',  
 'EXT_SOURCE_1',
 'EXT_SOURCE_2',
 'EXT_SOURCE_3',
 'FLAG_MODEL_PREDICTED',
 'SOCIODEMO_MODEL_PREDICTED',
 'TARGET']


def train_binary(X ,y, estimator, cv = 3, refit_all = True, verbose = True):
    """
    Creates a feature based on predictions from kfold
    """
    ls = list()
    result = {}
    kf = StratifiedKFold(n_splits = cv)
    k = 0
    for train_idx, test_idx in kf.split(X, y):
        k+=1
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = estimator.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)[:,1]
        if verbose:
            score = roc_auc_score(y_true=y_test, y_score=predictions)
            print(f"Score on test set for fold {k} is :{round(score,3)}")
        
        ls.append(predictions)
        
    feature = np.hstack(ls)
    
    if refit_all:
        model = estimator.fit(X, y)
        
        result["model"] = model
    
    result["feature"] = feature
    
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
        y = series
        lr = LinearRegression().fit(X, y)
    
        result["trend"] = lr.coef_[0]
        result["intercept"] = lr.intercept_
    except:
        result["trend"] = -1
        result["intercept"] = 1
    return result


def main(application_train, application_test, previous_application, bureau, bureau_balance, installments_payments, credit_card_balance, pos_history, return_X_y = False, verbose = True):
    """
    Generates a binary classification model

    """

    print(f"Processing bureau balance...")
    t0 = time()

    train_skid_curr = application_train["SK_ID_CURR"].unique().tolist()
    test_skid_curr = application_test["SK_ID_CURR"].unique().tolist()


    # For preprocessing
    bureau_balance["STATUS_NUMERIC"] = bureau_balance.STATUS.map({"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"X":None,"C":None})
    bureau_balance["EVER_DELINQUENT"] = np.where(bureau_balance.STATUS != '0',1,0)

    aux_c = bureau_balance[bureau_balance.STATUS == "0"].groupby("SK_ID_BUREAU").STATUS.size().to_frame().reset_index().rename(columns = {"STATUS":"TIMES_BUCKET_0"})
    aux_1 = bureau_balance[bureau_balance.STATUS == "1"].groupby("SK_ID_BUREAU").STATUS.size().to_frame().reset_index().rename(columns = {"STATUS":"TIMES_BUCKET_1"})
    aux_2 = bureau_balance[bureau_balance.STATUS == "2"].groupby("SK_ID_BUREAU").STATUS.size().to_frame().reset_index().rename(columns = {"STATUS":"TIMES_BUCKET_2"})
    aux_3 = bureau_balance[bureau_balance.STATUS == "3"].groupby("SK_ID_BUREAU").STATUS.size().to_frame().reset_index().rename(columns = {"STATUS":"TIMES_BUCKET_3"})
    aux_4 = bureau_balance[bureau_balance.STATUS == "4"].groupby("SK_ID_BUREAU").STATUS.size().to_frame().reset_index().rename(columns = {"STATUS":"TIMES_BUCKET_4"})
    aux_5 = bureau_balance[bureau_balance.STATUS == "5"].groupby("SK_ID_BUREAU").STATUS.size().to_frame().reset_index().rename(columns = {"STATUS":"TIMES_BUCKET_5"})
    aux_6 = bureau_balance[bureau_balance.STATUS == "6"].groupby("SK_ID_BUREAU").STATUS.size().to_frame().reset_index().rename(columns = {"STATUS":"TIMES_BUCKET_6"})

    
    # aux_ever_delinq = bureau_balance.groupby("SK_ID_BUREAU").STATUS_NUMERIC.apply(lambda x: get_ever_delinquent(x)).to_frame().reset_index().rename(columns = {"STATUS_NUMERIC":"EVER_DELINQUENT"})
    aux_ever_delinq = bureau_balance.groupby("SK_ID_BUREAU").EVER_DELINQUENT.max().to_frame().reset_index().rename(columns = {"STATUS":"EVER_DELINQUENT"})
    aux_ph = bureau_balance.groupby("SK_ID_BUREAU").STATUS_NUMERIC.mean().to_frame().reset_index().rename(columns = {"STATUS_NUMERIC":"PAYMENT_HISTORY"})
    aux_consecutive = bureau_balance.groupby("SK_ID_BUREAU").STATUS_NUMERIC.apply(lambda x:consecutive_values(x, 0)).to_frame().reset_index().rename(columns = {"STATUS_NUMERIC":"CONSECUTIVE_NO_DELINQ"})

    
    ldf_aux = [bureau,aux_c, aux_1, aux_2, aux_3, aux_4, aux_5, aux_6, aux_ph, aux_consecutive,aux_ever_delinq]

    bureau = reduce(lambda x, y: pd.merge(x, y, on = "SK_ID_BUREAU", how = "left"), ldf_aux)
    print(f"Bureau processed on: {round((time() - t0)/60,2)} minutes")

    print(f"Training stacked model with app features...")
    t0 = time()

    cont_prepro = Pipeline([("imputing", SimpleImputer(strategy = "constant", fill_value = -1)), ("preprocessing", StandardScaler())])
    cat_prepro = Pipeline([("imputing", SimpleImputer(strategy = "constant", fill_value = "missing")), ("encoding", OneHotEncoder(handle_unknown = "ignore"))])

    preprocessing = make_column_transformer((cont_prepro, selector(dtype_exclude = "object")), (cat_prepro,selector(dtype_include = "object")))

    dataset = application_train[flag_doc_feat].copy()

    estimator_flag = LogisticRegression(max_iter = 1000, class_weight = "balanced")
    estimator_social = Pipeline([("preprocessing", preprocessing),("model",LogisticRegression(class_weight= "balanced", max_iter= 1000))])


    X = application_train[flag_doc_feat].copy()
    y = application_train["TARGET"].copy()
    cust_behaiv_model = train_binary(X = X, y =y, estimator= estimator_flag, cv = 3)


    X = application_train[social_demo_feat].copy()
    y = application_train["TARGET"].copy()
    social_model = train_binary(X = X, y = y, estimator = estimator_social)

    application_train["FLAG_MODEL_PREDICTED"] = cust_behaiv_model["feature"]
    application_train["SOCIODEMO_MODEL_PREDICTED"] = social_model["feature"]

    application_test["FLAG_MODEL_PREDICTED"] = cust_behaiv_model["model"].predict_proba(application_test[flag_doc_feat])[:,1]
    application_test["SOCIODEMO_MODEL_PREDICTED"] = social_model["model"].predict_proba(application_test[social_demo_feat])[:,1]
    
    application_test["TARGET"] = None

    print(f"- application train instances: {application_train.shape[0]}")
    print(f"- application test instances: {application_test.shape[0]}")
    print(f"- concatenated dataset instances: {application_train.shape[0] + application_test.shape[0]}")

    application_data = pd.concat([application_train, application_test], ignore_index  = True)
    
    # application features
    application_data["ext_source_mean"] = (application_data["EXT_SOURCE_1"] + application_data["EXT_SOURCE_2"] + application_data["EXT_SOURCE_3"])/3
    application_data["entropy_ex_source"] = -(application_data["EXT_SOURCE_1"] * np.log(application_data["EXT_SOURCE_1"]) + application_data["EXT_SOURCE_2"] * np.log(application_data["EXT_SOURCE_2"]) + application_data["EXT_SOURCE_3"] * np.log(application_data["EXT_SOURCE_3"]))
    application_data["log_amt_credit_amt_income"] = np.log(application_data["AMT_CREDIT"]/application_data["AMT_INCOME_TOTAL"])
    application_data["log_amt_goods_price_amt_income"] = np.log(application_data["AMT_GOODS_PRICE"]/application_data["AMT_INCOME_TOTAL"])


    sk_id = application_data["SK_ID_CURR"].unique().tolist()

    data = pd.DataFrame(index = sk_id)

    print(f"Model based features on: {round((time() - t0)/60,2)} minutes")
    print("Initial dataframe info:")
    print(data.info())

    print(f"Creating features...")
    T0 = time()
    t0 = time()
    # total accounts
    data["total_accounts"] = bureau.groupby("SK_ID_CURR").size()
    # Open and closed accounts
    data["active_accounts"] = bureau.groupby("SK_ID_CURR").apply(lambda x: (x["CREDIT_ACTIVE"] == "Active").sum())
    
    data["cloased_accounts"] = bureau.groupby("SK_ID_CURR").apply(lambda x: (x["CREDIT_ACTIVE"] == "Closed").sum())
    data["max_amt_overdue"] = bureau.groupby("SK_ID_CURR").AMT_CREDIT_MAX_OVERDUE.max()
    data["mean_amt_overdue"] = bureau.groupby("SK_ID_CURR").AMT_CREDIT_MAX_OVERDUE.mean()
    data["mean_amt_credit_debt"]=bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].mean()
    data["sum_amt_credit_sum"]=bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].sum()
    data["sum_amt_credit_debt"]=bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].sum()
    data["mean_amt_credit_limit"]=bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].mean()
    data["sum_amt_credit_limit"]=bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].sum()
    data["sum_amt_credit_overdue"]=bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"].sum()
    print(f"First batch of features on: {round((time() - t0)/60,2)} minutes")
    
    t0 = time()
    # dividir entre AMT_CREDIT_SUM
    data["total_annuity"] = bureau.groupby("SK_ID_CURR").AMT_ANNUITY.sum()
    # Percentage open accounts
    data["active_prem_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(premium_loans)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.size()
    data["active_wk_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(working_capital_loans)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.size()
    data["active_bank_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(bank_credits)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.size()
    data["active_credit_cards"] = bureau[(bureau["CREDIT_TYPE"].isin(credit_cards)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.size()
    data["active_other_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(others)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.size()
    # Percentage closed accounts
    data["closed_prem_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(premium_loans)) & (bureau["CREDIT_ACTIVE"] == "Closed")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.size()
    data["closed_wk_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(working_capital_loans)) & (bureau["CREDIT_ACTIVE"] == "Closed")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.size()
    data["closed_bank_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(bank_credits)) & (bureau["CREDIT_ACTIVE"] == "Closed")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.size()
    data["closed_credit_cards"] = bureau[(bureau["CREDIT_TYPE"].isin(credit_cards)) & (bureau["CREDIT_ACTIVE"] == "Closed")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.size()
    data["closed_other_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(others)) & (bureau["CREDIT_ACTIVE"] == "Closed")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.size()
    # Saldo en $$ en cuentas abiertas
    data["sum_debt_active_prem_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(premium_loans)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.sum()
    data["sum_debt_active_wk_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(working_capital_loans)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.sum()
    data["sum_debt_active_bank_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(bank_credits)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.sum()
    data["sum_debt_active_credit_cards"] = bureau[(bureau["CREDIT_TYPE"].isin(credit_cards)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.sum()
    data["sum_debt_active_other_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(others)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.sum()
    # Trend Saldo en $$ en cuentas abiertas
    data["trend_debt_active_credits"] = bureau[(bureau["CREDIT_ACTIVE"] == "Active")].sort_values(by = ["SK_ID_CURR", "DAYS_CREDIT"]).groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.fillna(-1).apply(lambda x: get_linear_regression(x)["trend"])
    # Max en $$ en cuentas cerradas
    data["max_debt_closed_premium_loans"] = bureau[(bureau["CREDIT_TYPE"].isin(premium_loans)) & (bureau["CREDIT_ACTIVE"] == "Closed")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.max()
    data["max_debt_closed_wk_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(working_capital_loans)) & (bureau["CREDIT_ACTIVE"] == "Closed")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.max()
    data["max_debt_closed_bank_credits"] = bureau[(bureau["CREDIT_TYPE"].isin(bank_credits)) & (bureau["CREDIT_ACTIVE"] == "Closed")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.max()
    data["max_debt_closed_credit_card"] = bureau[(bureau["CREDIT_TYPE"].isin(credit_cards)) & (bureau["CREDIT_ACTIVE"] == "Closed")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.max()
    data["max_debt_closed_debt_others"] = bureau[(bureau["CREDIT_TYPE"].isin(others)) & (bureau["CREDIT_ACTIVE"] == "Closed")].groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.max()
    # div
    data["sum_amt_credit_limit_credit_cards_open"]=bureau[(bureau["CREDIT_TYPE"].isin(credit_cards)) & (bureau["CREDIT_ACTIVE"] == "Active")].groupby("SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].sum()
    data["perc_util_revolving"] = data["sum_debt_active_credit_cards"]/(data["sum_amt_credit_limit_credit_cards_open"] + 1)
    data["credit_debt_div_lim_cred"] = data["sum_amt_credit_debt"]/(data["sum_amt_credit_limit"] + 1)
    data["credit_lim_div_cred_overdue"] = data["sum_amt_credit_limit"]/(data["sum_amt_credit_overdue"] + 1)

    data.drop(["sum_amt_credit_sum", "sum_amt_credit_limit", "sum_amt_credit_limit_credit_cards_open"], axis = 1, inplace = True)
    # Max cred lim non defaulted
    data["max_cred_lim_non_delinq"] = bureau.groupby("SK_ID_CURR")[["EVER_DELINQUENT", "AMT_CREDIT_SUM_LIMIT"]].apply(lambda x: x[(x["EVER_DELINQUENT"] == 0)]["AMT_CREDIT_SUM_LIMIT"].max())
    # paymet history
    data["payment_history"] = bureau.groupby("SK_ID_CURR")["PAYMENT_HISTORY"].mean()
    data["mean_consecutive_non_delinq"] = bureau.groupby("SK_ID_CURR")["CONSECUTIVE_NO_DELINQ"].mean()
    data["max_consecutive_non_delinq"] = bureau.groupby("SK_ID_CURR")["CONSECUTIVE_NO_DELINQ"].max()
    print(f"Second batch of features on: {round((time() - t0)/60,2)} minutes")
    
    t0 = time()
    # Number of times in bucket x
    data["times_bucket_1"] = bureau.groupby("SK_ID_CURR")["TIMES_BUCKET_1"].sum()
    data["times_bucket_2"] = bureau.groupby("SK_ID_CURR")["TIMES_BUCKET_2"].sum()
    data["times_bucket_3"] = bureau.groupby("SK_ID_CURR")["TIMES_BUCKET_3"].sum()
    data["times_bucket_4"] = bureau.groupby("SK_ID_CURR")["TIMES_BUCKET_4"].sum()
    data["times_bucket_5"] = bureau.groupby("SK_ID_CURR")["TIMES_BUCKET_5"].sum()
    data["times_bucket_6"] = bureau.groupby("SK_ID_CURR")["TIMES_BUCKET_6"].sum()
    # Antiguedad meses
    data["antiguedad_media_dias"] = bureau.groupby("SK_ID_CURR")["DAYS_CREDIT"].mean()
    data["antiguedad_media_dias_closed_accounts"] = bureau.groupby("SK_ID_CURR")["DAYS_CREDIT_ENDDATE"].mean()
    data["antiguedad_media_dias_open_accounts"] = bureau.groupby("SK_ID_CURR")["DAYS_ENDDATE_FACT"].mean()
    data["sum_prolong_days"] = bureau.groupby("SK_ID_CURR")["CNT_CREDIT_PROLONG"].sum()
    # Closed accounts last x days
    data["closed_accounts_last12m"] = bureau[bureau["DAYS_ENDDATE_FACT"] >= -365].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["closed_accounts_last9m"] = bureau[bureau["DAYS_ENDDATE_FACT"] >= -270].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["closed_accounts_last6m"] = bureau[bureau["DAYS_ENDDATE_FACT"] >= -180].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["closed_accounts_last3m"] = bureau[bureau["DAYS_ENDDATE_FACT"] >= -90].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["closed_accounts_last2m"] = bureau[bureau["DAYS_ENDDATE_FACT"] >= -60].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["closed_accounts_last1m"] = bureau[bureau["DAYS_ENDDATE_FACT"] >= -30].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    # Open credit last x days
    data["open_accounts_last12m"] = bureau[bureau["DAYS_CREDIT"] >= -365].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["open_accounts_last9m"] = bureau[bureau["DAYS_CREDIT"] >= -270].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["open_accounts_last6m"] = bureau[bureau["DAYS_CREDIT"] >= -180].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["open_accounts_last3m"] = bureau[bureau["DAYS_CREDIT"] >= -90].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["open_accounts_last2m"] = bureau[bureau["DAYS_CREDIT"] >= -60].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["open_accounts_last1m"] = bureau[bureau["DAYS_CREDIT"] >= -30].groupby("SK_ID_CURR")["CREDIT_TYPE"].size()
    data["n_unique_credit_types"] = bureau.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique()
    # Trigonometric features
    data["cosine_cred_sum"] = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].apply(lambda x: (2 * np.pi  *np.cos(x)))
    data["sine_cred_sum"] = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].apply(lambda x: (2 * np.pi  *np.sin(x)))
    data["cosine_cred_sum"] = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].apply(lambda x: (2 * np.pi  *np.cos(x)))
    data["sine_cred_sum"] = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].apply(lambda x: (2 * np.pi  *np.sin(x)))
    data["cosine_cred_sum"] = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].apply(lambda x: (2 * np.pi  *np.cos(x)))
    data["sine_cred_sum"] = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].apply(lambda x: (2 * np.pi  *np.sin(x)))
    
    # entropy of features
    # data["credit_type_entropy"] = bureau.groupby("SK_ID_CURR")["CREDIT_TYPE"].apply(lambda x: get_entropy(x))
    # data["amt_cred_sum_entropy"] = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].apply(lambda x: get_entropy(x, categorical = False))
    # data["amt_sum_debt_entropy"] = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].apply(lambda x: get_entropy(x, categorical = False))
    # data["amt_cred_limit_entropy"] = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].apply(lambda x: get_entropy(x, categorical = False))
    print(f"Third batch of features on: {round((time() - t0)/60,2)} minutes")
    
    t0 = time()
    # previous applications features
    data["sum_prev_applications"] = previous_application.groupby("SK_ID_CURR").SK_ID_PREV.size()
    data["mean_amt_prev_applications"] = previous_application.groupby("SK_ID_CURR").AMT_APPLICATION.mean()
    data["mean_amt_cred_prev_applications"] = previous_application.groupby("SK_ID_CURR").AMT_CREDIT.mean()
    data["mean_amt_annuity_prev_applications"] = previous_application.groupby("SK_ID_CURR").AMT_ANNUITY.mean()
    data["sum_amt_downpayment_prev_applications"] = previous_application.groupby("SK_ID_CURR").AMT_DOWN_PAYMENT.sum()
    data["mean_amt_goodsprice_prev_applications"] = previous_application.groupby("SK_ID_CURR").AMT_GOODS_PRICE.mean()
    data["mean_days_last_due_prev_applications"] = previous_application.groupby("SK_ID_CURR").DAYS_LAST_DUE.mean()
    data["mean_days_first_due_prev_applications"] = previous_application.groupby("SK_ID_CURR").DAYS_FIRST_DUE.mean()
    data["mean_term_prev_applications"] = previous_application.groupby("SK_ID_CURR").CNT_PAYMENT.mean()
    data["total_previous_revolving_credits"] = previous_application[previous_application["NAME_CONTRACT_TYPE"] == "Revolving loans"].groupby("SK_ID_CURR").SK_ID_PREV.size()
    print(f"Fourth batch of features on: {round((time() - t0)/60,2)} minutes")
    
    data["mean_amt_credit_previous_revolving_credits"] = previous_application[previous_application["NAME_CONTRACT_TYPE"] == "Revolving loans"].groupby("SK_ID_CURR")["AMT_CREDIT"].mean()
    data["mean_amt_downpayment_previous_revolving_credits"] = previous_application[previous_application["NAME_CONTRACT_TYPE"] == "Revolving loans"].groupby("SK_ID_CURR")["AMT_DOWN_PAYMENT"].mean()
    # pos features
    data["cnt_installments_pos_mean"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT.sum().groupby("SK_ID_CURR").mean()
    data["cnt_installments_pos_max"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT.sum().groupby("SK_ID_CURR").max()
    data["cnt_installments_pos_total"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT.sum().groupby("SK_ID_CURR").sum()
    data["cnt_installments_pos_trend"] =pos_history.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()
    
    data["cnt_installments_fut_pos_mean"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT_FUTURE.sum().groupby("SK_ID_CURR").mean()
    data["cnt_installments_fut_pos_max"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT_FUTURE.sum().groupby("SK_ID_CURR").max()
    data["cnt_installments_fut_pos_total"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT_FUTURE.sum().groupby("SK_ID_CURR").sum()
    data["cnt_installments_fut_pos_trend"] =pos_history.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT_FUTURE.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()
    data["cnt_installments_pos_minus_pos_f"] = data["cnt_installments_pos_total"] - data["cnt_installments_fut_pos_total"] 
    
    data["cnt_installments_dpd_pos_mean"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD.sum().groupby("SK_ID_CURR").mean()
    data["cnt_installments_dpd_pos_max"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD.sum().groupby("SK_ID_CURR").max()
    data["cnt_installments_dpd_pos_total"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD.sum().groupby("SK_ID_CURR").sum()
    data["cnt_installments_dpd_pos_trend"] =pos_history.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()
      
    pos_history["PAR_X_AT_Y"] = pos_history["SK_DPD"]/pos_history["MONTHS_BALANCE"]

    data["cnt_installments_par_pos_mean"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).PAR_X_AT_Y.sum().groupby("SK_ID_CURR").mean()
    data["cnt_installments_par_pos_max"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).PAR_X_AT_Y.sum().groupby("SK_ID_CURR").max()
    data["cnt_installments_par_pos_total"] =pos_history.groupby(["SK_ID_CURR", "SK_ID_PREV"]).PAR_X_AT_Y.sum().groupby("SK_ID_CURR").sum()
    data["cnt_installments_par_pos_trend"] =pos_history.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).PAR_X_AT_Y.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()
   

    # credit cards balance
    t0 = time()
    data["mean_credit_cards_months"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_BALANCE.size().groupby("SK_ID_CURR").mean()
    data["max_credit_cards_months"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_BALANCE.size().groupby("SK_ID_CURR").max()
    data["total_credit_cards_amt_balance"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_BALANCE.sum().groupby("SK_ID_CURR").sum()
    data["max_credit_cards_amt_balance"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_BALANCE.sum().groupby("SK_ID_CURR").max()
    data["mean_credit_cards_amt_balance"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_BALANCE.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_balance"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_BALANCE.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()
    print(f"Fifth batch of features on: {round((time() - t0)/60,2)} minutes")
    
    
    t0 = time()
    data["max_credit_cards_amt_balance"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_CREDIT_LIMIT_ACTUAL.sum().groupby("SK_ID_CURR").max()
    data["mean_credit_cards_amt_balance"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_CREDIT_LIMIT_ACTUAL.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_balance"] = credit_card_balance.sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_CREDIT_LIMIT_ACTUAL.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()
    data["max_credit_cards_amt_paymentcurrent"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_amt_paymentcurrent"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_CURRENT.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_amt_paymentcurrent"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_paymentcurrent"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_amt_paymentcurrent"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["mean_credit_cards_amt_paymentcurrent"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_paymentcurrent"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_amt_total_recieivable"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_TOTAL_RECEIVABLE.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_amt_total_recieivable"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_TOTAL_RECEIVABLE.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_amt_total_recieivable"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_TOTAL_RECEIVABLE.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_total_recieivable"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_TOTAL_RECEIVABLE.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_amt_min_regularity"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_INST_MIN_REGULARITY.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_amt_min_regularity"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_INST_MIN_REGULARITY.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_amt_min_regularity"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_INST_MIN_REGULARITY.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_min_regularity"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_INST_MIN_REGULARITY.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_amt_payment_total_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_TOTAL_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_amt_payment_total_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_TOTAL_CURRENT.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_amt_payment_total_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_TOTAL_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_payment_total_current"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT_TOTAL_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_amt_drawings_atm_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_ATM_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_amt_drawings_atm_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_ATM_CURRENT.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_amt_drawings_atm_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_ATM_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_drawings_atm_current"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_ATM_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_amt_drawings_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_amt_drawings_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_CURRENT.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_amt_drawings_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_drawings_current"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()
    print(f"Sixth batch of features on: {round((time() - t0)/60,2)} minutes")
    
    t0 = time()
    data["max_credit_cards_amt_drawings_other_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_OTHER_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_amt_drawings_other_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_OTHER_CURRENT.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_amt_drawings_other_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_OTHER_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_drawings_other_current"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_OTHER_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_amt_drawings_pos_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_POS_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_amt_drawings_pos_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_POS_CURRENT.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_amt_drawings_pos_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_POS_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_drawings_pos_total_current"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_DRAWINGS_POS_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_amt_recivable"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_RECIVABLE.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_amt_recivable"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_RECIVABLE.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_amt_recivable"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_RECIVABLE.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_amt_recivable"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_RECIVABLE.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_cnt_atm_drawings_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_ATM_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_cnt_atm_drawings_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_ATM_CURRENT.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_cnt_atm_drawings_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_ATM_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_cnt_atm_drawings_current"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_ATM_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_cnt_drawings_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_cnt_drawings_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_CURRENT.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_cnt_drawings_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_cnt_drawings_current"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_cnt_drawings_other_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_OTHER_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_cnt_drawings_other_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_OTHER_CURRENT.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_cnt_drawings_other_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_OTHER_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_cnt_drawings_other_current"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_OTHER_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_cnt_drawings_pos_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_POS_CURRENT.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_cnt_drawings_pos_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_POS_CURRENT.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_cnt_drawings_pos_current"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_POS_CURRENT.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_cnt_drawings_pos_current"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_DRAWINGS_POS_CURRENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_cnt_installment_mature_cum"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT_MATURE_CUM.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_cnt_installment_mature_cum"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT_MATURE_CUM.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_cnt_installment_mature_cum"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT_MATURE_CUM.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_cards_cnt_installment_mature_cum"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).CNT_INSTALMENT_MATURE_CUM.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    data["max_credit_cards_dpd_def"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD_DEF.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_dpd_def"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD_DEF.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_dpd_def"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD_DEF.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_dpd_def"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD_DEF.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()
    
    data["max_credit_cards_dpd"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD.sum().groupby("SK_ID_CURR").max()
    data["total_credit_cards_dpd"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD.sum().groupby("SK_ID_CURR").sum()
    data["mean_credit_cards_dpd"] = credit_card_balance.groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD.sum().groupby("SK_ID_CURR").mean()
    data["mean_trend_credit_dpd"] = credit_card_balance.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).SK_DPD.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()

    # interacciones credit card balance

    data["amt_paycurr_div_total_amt_balance"] = data["max_credit_cards_amt_paymentcurrent"]/data["total_credit_cards_amt_balance"]
    data["amt_total_balnce_div_total_recivable"] = data["total_credit_cards_amt_balance"]/data["total_credit_cards_amt_total_recieivable"]

    # installmet features
    data["amt_installments_max_amt"] =installments_payments.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_INSTALMENT.sum().groupby("SK_ID_CURR").max()
    data["amt_installments_total_amt"] =installments_payments.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_INSTALMENT.sum().groupby("SK_ID_CURR").sum()
    data["amt_installments_mean_amt"] =installments_payments.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_INSTALMENT.sum().groupby("SK_ID_CURR").mean()
    data["amt_installments_trend"] =installments_payments.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV","NUM_INSTALMENT_NUMBER"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_INSTALMENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()
    
    data["amt_pay_installments_max_amt"] =installments_payments.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT.sum().groupby("SK_ID_CURR").max()
    data["amt_pay_installments_total_amtl"] =installments_payments.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT.sum().groupby("SK_ID_CURR").sum()
    data["amt_pay_installments_mean_amt"] =installments_payments.groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_INSTALMENT.sum().groupby("SK_ID_CURR").mean()
    data["amt_pay_installments_trend"] =installments_payments.fillna(0).sort_values(by = ["SK_ID_CURR","SK_ID_PREV","NUM_INSTALMENT_NUMBER"]).groupby(["SK_ID_CURR", "SK_ID_PREV"]).AMT_PAYMENT.apply(lambda x: get_linear_regression(x)["trend"]).groupby("SK_ID_CURR").mean()
    

    data["installments_payment_vs_amt_installment"] = data["amt_pay_installments_total_amtl"]/data["amt_installments_total_amt"]
    print(f"Seventh batch of features on: {round((time() - t0)/60,2)} minutes")
    
    data.reset_index(inplace = True)
    data.rename(columns = {"index":"SK_ID_CURR"}, inplace = True)
    data = data.merge(application_data[main_application_feat], on = "SK_ID_CURR", how = "left")

    data["split"] = np.where(data.index.isin(train_skid_curr), "train", "test")
    print(f"Features on: {round((time() - T0)/60,2)} minutes")

    # data.to_pickle("../data/data.pkl")

    print("Successfully done!!!")
    data.columns = data.columns.str.lower()
    print(data.info())

    if return_X_y:
    	X = data.drop("target", axis = 1)
    	y = data["target"]

    	return X, y
    else:
    	return data