import pandas as pd
import numpy as np

# local imports
from src.config import logger

class ProcessCreditCardBalance():
    """
    """

    def __init__(self, credit_card_df):
        """
        """
        self.credit_card_df = credit_card_df
    
    def process(self):
        """
        """
        logger.info("Preprocessing dataframe...")
        self.credit_card_df.columns = self.credit_card_df.columns.str.lower()

        # Amount used from limit
        self.credit_card_df['limit_use'] = self.credit_card_df['amt_balance'] / self.credit_card_df['amt_credit_limit_actual']
        # Current payment / Min payment
        self.credit_card_df['payment_div_min'] = self.credit_card_df['amt_payment_current'] / self.credit_card_df['amt_inst_min_regularity']
        # Late payment <-- 'CARD_IS_DPD'
        self.credit_card_df['late_payment'] = self.credit_card_df['sk_dpd'].apply(lambda x: 1 if x > 0 else 0)
        # How much drawing of limit
        self.credit_card_df['drawing_limit_ratio'] = self.credit_card_df['amt_drawings_atm_current'] / self.credit_card_df['amt_credit_limit_actual']
        
        self.credit_card_df['card_is_dpd_under_120'] = self.credit_card_df['sk_dpd'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
        self.credit_card_df['card_is_dpd_over_120'] = self.credit_card_df['sk_dpd'].apply(lambda x: 1 if x >= 120 else 0)

        return self.credit_card_df
