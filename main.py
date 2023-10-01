import pandas as pd
import numpy as np

import sys

sys.path.append("../")

# local imports
from src.config import logger
from src.default_model import DefaultModel
from utils.functions__utils import load_csv_files


def main(
    data_directory: str = None,
    save_data_path: str = None,
    save_estimator_path: str = None,
    apply_shap: bool = False,
):
    """
    Main function for executing the default model pipeline.

    Parameters:
        data_directory (str, optional): Path to the data directory. Default is None.
        model_path (str, optional): Path to a trained model. Default is None.
        train_model (bool, optional): Whether to train the model. Default is False.
        apply_shap (bool, optional): Whether to apply SHAP values during inference. Default is False.

    Returns:
        pd.DataFrame: DataFrame containing the model's predictions or inferences.

    This function orchestrates the execution of the default model pipeline, including data preparation,
    model training, and making inferences. It allows for specifying data and model paths, controlling
    whether to train the model, and applying SHAP values during inference.
    """

    logger.info("loading the datasets...")
    loaded_data = load_csv_files(data_directory)

    application_train_df = loaded_data["application_train"]
    application_test_df = loaded_data["application_test"]
    bureau_balance_df = loaded_data["bureau_balance"]
    bureau_df = loaded_data["bureau"]
    credit_card_balance_df = loaded_data["credit_card_balance"]
    installments_payments_df = loaded_data["installments_payments"]
    pos_cash_balance_df = loaded_data["pos_cash_balance"]
    previous_application_df = loaded_data["previous_application"]
    logger.info("datasets loaded.")

    default_model = DefaultModel(
        application_train_df,
        application_test_df,
        bureau_balance_df,
        bureau_df,
        credit_card_balance_df,
        installments_payments_df,
        pos_cash_balance_df,
        previous_application_df,
    )

    logger.info("creating input dataset...")
    input_dataset = default_model.create_input_dataset(verbose=False)
    logger.info("input dataset created.")

    bst = default_model.train_predict_fn(save_estimator_path)

    output_df = default_model.make_inference(
        save_data_path=save_data_path, apply_shap=apply_shap
    )

    return output_df


if __name__ == "__main__":
    output_df = main(
        "data/", save_data_path=f"data/submissions/final_submission_20230930.pkl"
    )
