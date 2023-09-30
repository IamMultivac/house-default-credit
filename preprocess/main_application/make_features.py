import pandas as pd
import numpy as np
from time import time
import sys

from lightgbm import LGBMClassifier as lgbm

from functools import reduce

merge_fn = lambda x,y:pd.merge(x,y, on = space_column, how = "inner")

agg_fns = ["min","max","mean","std"]

sys.path.append("../")

# local imports
from preprocess.main_application.process import ProcessMainApplication
from utils.functions__utils import train_binary
from src.config import logger
from src.learner_params import target_column, space_column, prediction_column, base_learners_params
from utils.features_lists import application_base_features, application_categorical_columns, application_numerical_columns

prefix = "main_application"

bst = lgbm(**base_learners_params)

def main(application_train_df:pd.DataFrame,
         application_test_df:pd.DataFrame,
         verbose:bool = True
        ):
    
    
    """
    Creates all the features
    """
    
    T0 = time()
    application_df = pd.concat([application_train_df, application_test_df], ignore_index = True)

    application = ProcessMainApplication(application_df)

    frame = application.process()
    logger.info("Creating features...")
    
    frame["ext_source_mean"] = (frame["ext_source_1"] + frame["ext_source_2"] + frame["ext_source_3"]) / 3
    frame["ext_source_entropy"] = -(frame["ext_source_1"] * np.log1p(frame["ext_source_1"]) + frame["ext_source_2"] * np.log1p(frame["ext_source_2"]) + frame["ext_source_3"] * np.log1p(frame["ext_source_1"]))
    frame["log_amt_credit_amt_income"] = np.log1p(frame["amt_credit"] / frame["amt_income_total"])
    frame["log_amt_goods_price_amt_income"] = np.log(frame["amt_goods_price"] / frame["amt_income_total"])
    frame['credit_to_annuity_ratio'] = frame['amt_credit'] / frame['amt_annuity']
    frame['credit_to_goods_ratio'] = frame['amt_credit'] / frame['amt_goods_price']
    frame['income_per_child'] = frame['amt_income_total'] / frame['cnt_children']
    frame['payment_rate'] = frame['amt_annuity'] / frame['amt_credit']

    frame['children_ratio'] = frame['cnt_children'] / frame['cnt_fam_members']
    frame['credit_to_income_ratio'] = frame['amt_credit'] / frame['amt_income_total']
    frame['income_per_child'] = frame['amt_income_total'] / (1 + frame['cnt_children'])
    frame['phone_to_birth_ratio'] = frame['days_last_phone_change'] / frame['days_birth']
    frame['phone_to_employ_ratio'] = frame['days_last_phone_change'] / frame['days_employed']
    

    
    # Flag_document features - count and kurtosis
    docs = [f for f in frame.columns if 'flag_doc' in f]
    frame['document_percentage'] = frame[docs].mean(axis=1)
    
    # Some simple new features (percentages)
    frame['days_employed_perc'] = frame['days_employed'] / frame['days_birth']
    frame['income_credit_percentage'] = frame['amt_income_total'] / frame['amt_credit']
    frame['income_per_person'] = frame['amt_income_total'] / frame['cnt_fam_members']
    frame['annuity_income_perc'] = frame['amt_annuity'] / frame['amt_income_total']

    
    
    # Income ratios
    frame['income_to_employed_ratio'] = frame['amt_income_total'] / frame['days_employed']
    frame['income_to_birth_ratio'] = frame['amt_income_total'] / frame['days_birth']
    
    # Time ratios
    frame['id_to_birth_ratio'] = frame['days_id_publish'] / frame['days_birth']
    frame['car_to_birth_ratio'] = frame['own_car_age'] / frame['days_birth']
    frame['car_to_employed_ratio'] = frame['own_car_age'] / frame['days_employed']
    frame['phone_to_birth_ratio'] = frame['days_last_phone_change'] / frame['days_birth']
    
    # ext_source_x FEATURE
    frame['ext_source_std'] = frame[['ext_source_1', 'ext_source_2', 'ext_source_3']].std(axis=1)
    frame['ext_source_min'] = frame[['ext_source_1', 'ext_source_2', 'ext_source_3']].min(axis=1)
    frame['ext_source_max'] = frame[['ext_source_1', 'ext_source_2', 'ext_source_3']].max(axis=1)   
    frame['ext_source_avg_null'] =frame[['ext_source_1', 'ext_source_2', 'ext_source_3']].isnull().mean(axis = 1)
    
    frame['score1_to_birth_ratio'] = frame['ext_source_1'] / (frame['days_birth'] / 365.25)
    frame['score2_to_birth_ratio'] = frame['ext_source_2'] / (frame['days_birth'] / 365.25)
    frame['score3_to_birth_ratio'] = frame['ext_source_3'] / (frame['days_birth'] / 365.25)
    frame['score1_to_employ_ratio'] = frame['ext_source_1'] / (frame['days_employed'] / 365.25)
    frame['ext_source_2*ext_source_3*days_birth'] = frame['ext_source_1'] * frame['ext_source_2'] * frame['days_birth']
    frame['score1_to_fam_cnt_ratio'] = frame['ext_source_1'] / frame['cnt_fam_members']
    frame['score1_to_goods_ratio'] = frame['ext_source_1'] / frame['amt_goods_price']
    frame['score1_to_credit_ratio'] = frame['ext_source_1'] / frame['amt_credit']
    frame['score1_to_score2_ratio'] = frame['ext_source_1'] / frame['ext_source_2']
    frame['score1_to_score3_ratio'] = frame['ext_source_1'] / frame['ext_source_3']
    frame['score2_to_credit_ratio'] = frame['ext_source_2'] / frame['amt_credit']
    frame['score2_to_region_rating_ratio'] = frame['ext_source_2'] / frame['region_rating_client']
    frame['score2_to_city_rating_ratio'] = frame['ext_source_2'] / frame['region_rating_client_w_city']
    frame['score2_to_pop_ratio'] = frame['ext_source_2'] / frame['region_population_relative']
    frame['score2_to_phone_change_ratio'] = frame['ext_source_2'] / frame['days_last_phone_change']
    frame['ext_source_1*ext_source_2'] = frame['ext_source_1'] * frame['ext_source_2']
    frame['ext_source_1*ext_source_3'] = frame['ext_source_1'] * frame['ext_source_3']
    frame['ext_source_2*ext_source_3'] = frame['ext_source_2'] * frame['ext_source_3']
    frame['ext_source_1*days_employed'] = frame['ext_source_1'] * frame['days_employed']
    frame['ext_source_2*days_employed'] = frame['ext_source_2'] * frame['days_employed']
    frame['ext_source_3*days_employed'] = frame['ext_source_3'] * frame['days_employed']
    
    # amt_income_total : income
    # cnt_fam_members  : the number of family members
    frame['goods_income_ratio'] = frame['amt_goods_price'] / frame['amt_income_total']
    
    # days_employed : How many days before the application the person started current employment
    frame['income_employed_ratio'] = frame['amt_income_total'] / frame['days_employed']
    frame['is_unemployed'] = (frame['days_employed'] == 365243).astype(int)
    
    frame['amt_income_total_12_amt_annuity_ratio'] = frame['amt_income_total'] / 12. - frame['amt_annuity']
    frame['income_to_employ_ratio'] = frame['amt_income_total'] / frame['days_employed']
    frame['days_last_phone_change_days_employed_ratio'] = frame['days_last_phone_change'] / frame['days_employed']
    frame['days_employed_days_birth_diff'] = frame['days_employed'] - frame['days_birth']

    logger.info("Training a base learner...")
    application_train_df.columns = application_train_df.columns.str.lower()
    application_test_df.columns = application_test_df.columns.str.lower()
    base_learner_logs = train_binary(merge_fn(frame, application_train_df[[space_column]]),
                                       application_base_features,
                                       target_column,
                                       bst
                                      )
    base_learner_train= base_learner_logs["data"]
    
    base_learner_test = base_learner_logs["p"](merge_fn(frame, application_test_df[[space_column]]))

    base_learner_df = pd.concat([base_learner_train, base_learner_test], ignore_index = True)[[space_column, prediction_column]]

    for cat in application_categorical_columns:
        for num in application_numerical_columns:
            for agg_fn in agg_fns:
                _mapa = dict(application_train_df.groupby(cat)[num].agg(agg_fn))
                application_train_df.loc[:,f"category_encoded_{cat}_{agg_fn}_{num}"] = application_train_df[cat].map(_mapa)
                application_test_df.loc[:,f"category_encoded_{cat}_{agg_fn}_{num}"] = application_test_df[cat].map(_mapa)

    aux_columns = [x for x in application_train_df.columns if "category_encoded" in x] + [space_column]

    aux_application_df = pd.concat([application_train_df, application_test_df], ignore_index = True)[aux_columns]

    frame = pd.merge(frame, aux_application_df, on = space_column)

    frame = pd.merge(frame, base_learner_df, on = space_column, how = "outer")

    frame.replace(np.inf,np.NaN, inplace = True)
    frame.replace(-np.inf,np.NaN, inplace = True)

    frame.columns = [f"{prefix}__{x}" for x in frame.columns.tolist()]

    frame = frame.rename(columns = {f"{prefix}__{space_column}":space_column,f"{prefix}__{target_column}":target_column})

    logger.info(f"Successfully created featureset of length: {len(frame)} in: {((time() - T0) / 60):.2f} minutes")

    if verbose:
        frame.info(verbose = True, show_counts = True)
    
    
    return frame
    
             




    