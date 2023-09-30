import pandas as pd
import numpy as np

# local imports
from src.config import logger

class ProcessInstallments():
    """
    """

    def __init__(self, installments_payments_df, main_application_df):
        """
        """
    
        self.installments_payments_df = installments_payments_df
        self.main_application_df = main_application_df
    
    def process(self):
        """
        """
        logger.info("Preprocessing dataframe...")
        self.main_application_df.columns = self.main_application_df.columns.str.lower()
        self.installments_payments_df.columns = self.installments_payments_df.columns.str.lower()

        cols = ["sk_id_curr", "amt_credit"]
        self.main_application_df = self.main_application_df.loc[:,cols]

        frame = pd.merge(self.installments_payments_df, self.main_application_df, on = "sk_id_curr", how = "left")
        frame.loc[:,"amt_payment"] = frame.loc[:,"amt_payment"].fillna(0)

        
        frame["paid_over_amount"] = (frame.loc[:,"amt_payment"] - frame.loc[:,"amt_instalment"])

        frame['payment_difference_credit'] = frame['amt_instalment'] - frame['amt_credit']
        frame['payment_ratio_credit'] = frame['amt_instalment'] / frame['amt_credit']
        frame['paid_over'] = (frame['paid_over_amount'] > 0).astype(int)
        
        # Percentage and difference paid in each installment (amount paid and installment value)
        frame['payment_perc'] = frame['amt_payment'] / frame['amt_instalment']
        frame['payment_diff'] = frame['amt_instalment'] - frame['amt_payment']
        
        # Days past due and days before due (no negative values)
        frame['dpd_diff'] = frame['days_entry_payment'] - frame['days_instalment']
        frame['dbd_diff'] = frame['days_instalment'] - frame['days_entry_payment']
        frame['dpd'] = frame['dpd_diff'].apply(lambda x: x if x > 0 else 0)
        frame['dbd'] = frame['dbd_diff'].apply(lambda x: x if x > 0 else 0)
        
        # Flag late payment
        frame['late_payment'] = frame['dbd'].apply(lambda x: 1 if x > 0 else 0)
        frame['instalment_payment_ratio'] = frame['amt_payment'] / frame['amt_instalment']
        frame['late_payment_ratio'] = frame.apply(lambda x: x['instalment_payment_ratio'] if x['late_payment'] == 1 else 0, axis=1)
        
        # Flag late payments that have a significant amount
        frame['significant_late_payment'] = frame['late_payment_ratio'].apply(lambda x: 1 if x > 0.05 else 0)

        return frame


        