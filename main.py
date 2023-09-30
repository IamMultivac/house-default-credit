import pandas as pd
import numpy as np

import sys

sys.path.append("../")

# local imports
from src.default_model import DefaultModel


def main(
    data_directory: str = None,
    model_path: str = None,
    train_model: bool = False,
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
    default_model = DefaultModel(
        application_train_df,
        application_test_df,
        bureau_balance_df,
        bureau_df,
        installments_payments_df,
        pos_cash_balance_df,
        previous_application_df,
    )

    input_dataset = default_model.create_input_dataset(verbose=False)

    if train_model:
        _ = default_model.train_predict_fn(save_estimator_path)

    output_df = default_model.make_inference(model_path=model_path, apply_shap=apply_shap)

    return output_df


if __name__ == "__main__":
    output_df = main("data/")
