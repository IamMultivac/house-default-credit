import joblib
import pandas as pd
import numpy as np

from functools import reduce

import sys

sys.path.append("../")

# local imports
from preprocess.bureau.make_features import main as make_bureau_features
from preprocess.credit_card_balance.make_features import (
    main as make_credit_card_features,
)
from preprocess.installments.make_features import main as make_installments_features
from preprocess.pos_cash.make_features import main as make_pos_features
from preprocess.previous_application.make_features import (
    main as make_previous_application_features,
)
from preprocess.main_application.make_features import main as make_application_features

from src.learner_params import (
    space_column,
    target_column,
    prediction_column,
    MODEL_PARAMS as params,
)

from utils.functions__training import lgbm_classification_learner

from config import logger


class DefaultModel:
    """
    DefaultModel is a class for creating an input dataset, training a model, and making inferences.

    Parameters:
        application_train_df (pd.DataFrame): DataFrame containing training data.
        application_test_df (pd.DataFrame): DataFrame containing testing data.
        bureau_balance_df (pd.DataFrame): DataFrame containing bureau balance data.
        bureau_df (pd.DataFrame): DataFrame containing bureau data.
        installments_payments_df (pd.DataFrame): DataFrame containing installments payments data.
        pos_cash_balance_df (pd.DataFrame): DataFrame containing POS cash balance data.
        previous_application_df (pd.DataFrame): DataFrame containing previous application data.
    """

    def __init__(
        self,
        application_train_df,
        application_test_df,
        bureau_balance_df,
        bureau_df,
        installments_payments_df,
        pos_cash_balance_df,
        previous_application_df,
    ):
        """ """
        self.application_train_df = application_train_df
        self.application_test_df = application_test_df
        self.application_test_df = application_test_df
        self.bureau_balance_df = bureau_balance_df
        self.bureau_df = bureau_df
        self.installments_payments_df = installments_payments_df
        self.pos_cash_balance_df = pos_cash_balance_df
        self.previous_application_df = previous_application_df

    def create_input_dataset(self, verbose: bool = False):
        """
        Create an input dataset by processing and merging multiple DataFrames.

        Parameters:
            verbose (bool, optional): Whether to display verbose output. Default is False.
        """

        bureau_features_df = make_bureau_features(
            self.bureau_df,
            self.bureau_balance_df,
            self.application_train_df,
            self.application_test_df,
        )

        credit_card_features_df = make_credit_card_features(
            self.credit_card_balance_df,
            self.application_train_df,
            self.application_test_df,
        )
        installments_features_df = make_installments_features(
            self.installments_payments_df, self.application_df
        )

        pos_features_df = make_pos_features(
            self.pos_cash_balance_df,
            self.application_train_df,
            self.application_test_df,
        )
        previous_application_features_df = make_previous_application_features(
            self.previous_application_df,
            self.application_train_df,
            self.application_test_df,
        )
        application_features_df = make_application_features(
            self.application_train_df, self.application_test_df
        )

        ldf = [
            bureau_features_df,
            credit_card_features_df,
            installments_features_df,
            pos_features_df,
            previous_application_features_df,
            application_features_df,
        ]

        self.frame = reduce(
            lambda x, y: pd.merge(x, y, on=space_column, how="inner"), ldf
        )

        return self.frame

    def train_predict_fn(self, save_estimator_path: str = None):
        """
        Train a model and save the trained estimator to a specified path.

        Parameters:
            save_estimator_path (str, optional): Path to save the trained estimator. Default is None.
        """
        self.bst, _, logs = lgbm_classification_learner(
            tmp_train_df,
            features=features,
            target=target_column,
            learning_rate=params["learner_params"]["learning_rate"],
            num_estimators=params["learner_params"]["n_estimators"],
            extra_params=params["learner_params"]["extra_params"],
        )

        with open(save_estimator_path, "wb") as context:
            cp.dump(bst, context)

    def make_inference(self, model_path: str = None, apply_shap: bool = False):
        """
        Make inferences using a trained model.

        Parameters:
            model_path (str, optional): Path to the trained model. Default is None.
            apply_shap (bool, optional): Whether to apply SHAP values during inference. Default is False.
        """
        if model_path:
            predict_fn = joblib.load(model_path)
            predictions = predict_fn["predict_fn"](self.frame, apply_shap=apply_shap)
        else:
            predictions = self.bs(self.frame, apply_shap=apply_shap)

        return predictions
